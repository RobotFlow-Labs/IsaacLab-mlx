# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native ANYmal-C flat locomotion task and MLX training helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from isaaclab.backends.runtime import (
    ArticulationCapabilities,
    MacSimBackend,
    SimBackendContract,
    SimCapabilities,
    resolve_runtime_selection,
    set_runtime_selection,
)
from isaaclab.utils.configclass import configclass

from .contacts import BatchedContactSensorState
from .env_cfgs import MacAnymalCFlatEnvCfg
from .locomotion import (
    action_rate_l2,
    base_contact_termination,
    feet_air_time_reward,
    flat_orientation_l2,
    track_linear_velocity_xy_exp,
    track_yaw_rate_z_exp,
    undesired_contacts,
)
from .quadcopter import _quat_conjugate, _quat_from_angular_velocity, _quat_multiply, _quat_normalize, _quat_rotate
from .reset_primitives import DeterministicResetSampler
from .state_primitives import BatchedArticulationState, BatchedRootState
from .terrain import MacPlaneTerrain

BODY_NAMES = (
    "base",
    "LF_FOOT",
    "RF_FOOT",
    "LH_FOOT",
    "RH_FOOT",
    "LF_THIGH",
    "RF_THIGH",
    "LH_THIGH",
    "RH_THIGH",
)
FOOT_BODY_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
THIGH_BODY_NAMES = ("LF_THIGH", "RF_THIGH", "LH_THIGH", "RH_THIGH")
BASE_BODY_NAMES = ("base",)
HIP_OFFSETS = mx.array(
    [
        [0.32, 0.20],
        [0.32, -0.20],
        [-0.32, 0.20],
        [-0.32, -0.20],
    ],
    dtype=mx.float32,
)
GAIT_PHASE_OFFSETS = mx.array([0.0, math.pi, math.pi, 0.0], dtype=mx.float32)
LOG_2_PI = math.log(2.0 * math.pi)


@configclass
class MacAnymalCTrainCfg:
    """Training configuration for the MLX ANYmal-C flat locomotion smoke path."""

    env: MacAnymalCFlatEnvCfg = MacAnymalCFlatEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.35
    checkpoint_path: str = "logs/mlx/anymal_c_flat_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix:
        return checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.json")
    return checkpoint_path.with_suffix(".json")


def _write_checkpoint_metadata(checkpoint_path: Path, cfg: MacAnymalCTrainCfg) -> Path:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    metadata = {
        "hidden_dim": cfg.hidden_dim,
        "observation_space": cfg.env.observation_space,
        "action_space": cfg.env.action_space,
        "action_std": cfg.action_std,
        "train_cfg": asdict(cfg),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _read_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class MacAnymalCFlatSimBackend(MacSimBackend):
    """A flat-terrain quadruped locomotion simulator for MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )
    contract = SimBackendContract(
        reset_signature="reset(soft: bool = False) -> None",
        step_signature="step(render: bool = True, update_fabric: bool = False) -> None",
        articulations=ArticulationCapabilities(
            joint_state_io=True,
            root_state_io=True,
            effort_targets=True,
            batched_views=True,
        ),
    )

    def __init__(self, cfg: MacAnymalCFlatEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.reset_sampler = reset_sampler or DeterministicResetSampler(cfg.seed)
        self.terrain = MacPlaneTerrain(
            cfg.num_envs,
            env_spacing=cfg.env_spacing,
            tile_size=cfg.terrain_tile_size,
        )
        self.root_state = BatchedRootState(cfg.num_envs, origin_grid=self.terrain.origin_grid)
        self.joint_state = BatchedArticulationState(cfg.num_envs, num_joints=cfg.action_space)
        self.default_joint_pos = mx.array(cfg.default_joint_pos, dtype=mx.float32).reshape((1, cfg.action_space))
        self.commands = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.joint_acc = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.applied_torque = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self._action_targets = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self._gait_phase = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self.contact_model = BatchedContactSensorState(
            cfg.num_envs,
            BODY_NAMES,
            foot_body_names=FOOT_BODY_NAMES,
            step_dt=cfg.sim_dt,
            history_length=cfg.contact_history_length,
            contact_margin=cfg.contact_margin,
            spring_stiffness=cfg.contact_stiffness,
            damping=cfg.contact_damping,
            force_threshold=cfg.contact_force_threshold,
        )
        self._last_body_pos_w = mx.zeros((cfg.num_envs, len(BODY_NAMES), 3), dtype=mx.float32)
        self.reset()

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    @property
    def env_origins(self) -> mx.array:
        return self.root_state.env_origins

    @property
    def root_pos_w(self) -> mx.array:
        return self.root_state.root_pos_w

    @property
    def root_lin_vel_b(self) -> mx.array:
        return self.root_state.root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> mx.array:
        return self.root_state.root_ang_vel_b

    @property
    def root_quat_w(self) -> mx.array:
        return self.root_state.root_quat_w

    @property
    def projected_gravity_b(self) -> mx.array:
        return self.root_state.projected_gravity_b

    def joint_state_view(self) -> tuple[mx.array, mx.array]:
        return self.get_joint_state(None)

    def set_commands(self, commands: Any) -> None:
        self.commands = mx.array(commands, dtype=mx.float32).reshape((self.num_envs, 3))

    def set_action_targets(self, actions: Any) -> None:
        self._action_targets = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space)), -1.0, 1.0)

    def reset(self, *, soft: bool = False) -> None:
        del soft
        self.reset_envs(list(range(self.num_envs)))

    def reset_envs(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)

        root_pos = self.terrain.spawn_positions(env_ids, (0.0, 0.0, self.cfg.default_root_height))
        root_pos[:, :2] = root_pos[:, :2] + self.reset_sampler.uniform((rows, 2), -0.05, 0.05)
        joint_pos = mx.broadcast_to(self.default_joint_pos, (rows, self.cfg.action_space)) + self.reset_sampler.uniform(
            (rows, self.cfg.action_space),
            -self.cfg.joint_reset_noise,
            self.cfg.joint_reset_noise,
        )
        joint_vel = mx.zeros((rows, self.cfg.action_space), dtype=mx.float32)

        self.root_state.reset_envs(
            env_ids,
            root_pos_w=root_pos,
            root_quat_w=(0.0, 0.0, 0.0, 1.0),
            root_lin_vel_b=0.0,
            root_ang_vel_b=0.0,
            projected_gravity_b=(0.0, 0.0, -1.0),
        )
        self.joint_state.reset_envs(env_ids, joint_pos=joint_pos, joint_vel=joint_vel, joint_effort_target=0.0)
        self._action_targets[ids] = 0.0
        self.joint_acc[ids] = 0.0
        self.applied_torque[ids] = 0.0
        self.commands[ids] = self.reset_sampler.uniform(
            (rows, 3),
            -self.cfg.command_scale,
            self.cfg.command_scale,
        )
        self._gait_phase[ids] = self.reset_sampler.uniform((rows,), 0.0, 2.0 * math.pi)
        self.contact_model.reset_envs(env_ids)
        body_pos_w = self._body_positions()[ids]
        self._last_body_pos_w[ids] = body_pos_w
        self.contact_model.update(body_pos_w, mx.zeros_like(body_pos_w), self.terrain, env_ids=env_ids)

    def _leg_extension(self, joint_pos: mx.array) -> mx.array:
        joint_pos = joint_pos.reshape((self.num_envs, 4, 3))
        hip_pitch = joint_pos[:, :, 1]
        knee = joint_pos[:, :, 2]
        extension = (
            0.20
            + 0.16 * mx.cos(hip_pitch)
            + 0.18 * mx.cos(hip_pitch + knee)
        )
        return mx.clip(extension, 0.22, 0.62)

    def _body_positions(self) -> mx.array:
        joint_pos = self.joint_state.joint_pos.reshape((self.num_envs, 4, 3))
        hip_abduction = joint_pos[:, :, 0]
        hip_pitch = joint_pos[:, :, 1]
        knee = joint_pos[:, :, 2]
        extension = self._leg_extension(self.joint_state.joint_pos)
        command_speed = mx.linalg.norm(self.commands[:, :2], axis=1, keepdims=True)
        phase = self._gait_phase.reshape((self.num_envs, 1)) + GAIT_PHASE_OFFSETS.reshape((1, 4))
        swing = mx.maximum(mx.sin(phase), 0.0) * (0.25 + command_speed)

        root_xy = self.root_state.root_pos_w[:, None, :2]
        root_z = self.root_state.root_pos_w[:, 2:3]
        step_x = 0.10 * self.commands[:, 0:1] * mx.cos(phase)
        step_y = 0.06 * self.commands[:, 1:2] * mx.sin(phase)
        foot_xy = root_xy + HIP_OFFSETS.reshape((1, 4, 2))
        foot_xy[:, :, 0] = foot_xy[:, :, 0] + step_x - 0.04 * mx.sin(hip_pitch)
        foot_xy[:, :, 1] = foot_xy[:, :, 1] + step_y + 0.04 * hip_abduction
        foot_z = root_z - extension + self.cfg.foot_clearance * swing - 0.02 * mx.tanh(knee)
        foot_pos = mx.concatenate([foot_xy, foot_z[:, :, None]], axis=-1)

        thigh_xy = root_xy + 0.55 * HIP_OFFSETS.reshape((1, 4, 2))
        thigh_xy[:, :, 1] = thigh_xy[:, :, 1] + 0.02 * hip_abduction
        thigh_z = root_z - 0.24 - 0.05 * mx.tanh(hip_pitch)
        thigh_pos = mx.concatenate([thigh_xy, thigh_z[:, :, None]], axis=-1)

        base_pos = self.root_state.root_pos_w[:, None, :]
        return mx.concatenate([base_pos, foot_pos, thigh_pos], axis=1)

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        dt = self.physics_dt

        desired_joint_pos = self.default_joint_pos + self.cfg.action_scale * self._action_targets
        joint_error = desired_joint_pos - self.joint_state.joint_pos
        self.applied_torque = self.cfg.joint_stiffness * joint_error - self.cfg.joint_damping * self.joint_state.joint_vel
        self.joint_acc = self.applied_torque / self.cfg.joint_inertia
        self.joint_state.joint_vel = mx.clip(self.joint_state.joint_vel + dt * self.joint_acc, -10.0, 10.0)
        self.joint_state.joint_pos = mx.clip(
            self.joint_state.joint_pos + dt * self.joint_state.joint_vel,
            mx.broadcast_to(self.default_joint_pos - 1.2, (self.num_envs, self.cfg.action_space)),
            mx.broadcast_to(self.default_joint_pos + 1.2, (self.num_envs, self.cfg.action_space)),
        )

        phase_speed = self.cfg.gait_frequency * (1.0 + 0.5 * mx.linalg.norm(self.commands[:, :2], axis=1))
        self._gait_phase = (self._gait_phase + dt * phase_speed * 2.0 * math.pi) % (2.0 * math.pi)

        body_pos_w = self._body_positions()
        body_vel_w = (body_pos_w - self._last_body_pos_w) / dt
        self._last_body_pos_w = body_pos_w
        self.contact_model.update(body_pos_w, body_vel_w, self.terrain)

        foot_ids = list(self.contact_model.foot_body_ids)
        support = self.contact_model.contact_mask[:, foot_ids].astype(mx.float32)
        support_ratio = mx.mean(support, axis=1)
        left_support = mx.mean(support[:, [0, 2]], axis=1)
        right_support = mx.mean(support[:, [1, 3]], axis=1)
        front_support = mx.mean(support[:, [0, 1]], axis=1)
        rear_support = mx.mean(support[:, [2, 3]], axis=1)

        grouped_actions = self._action_targets.reshape((self.num_envs, 4, 3))
        left_actions = mx.mean(grouped_actions[:, [0, 2], :], axis=(1, 2))
        right_actions = mx.mean(grouped_actions[:, [1, 3], :], axis=(1, 2))
        front_actions = mx.mean(grouped_actions[:, [0, 1], :], axis=(1, 2))
        rear_actions = mx.mean(grouped_actions[:, [2, 3], :], axis=(1, 2))

        lin_gain = self.cfg.command_tracking_gain * (0.35 + 0.65 * support_ratio)
        lin_xy = self.root_state.root_lin_vel_b[:, :2]
        lin_xy = lin_xy + dt * (
            lin_gain[:, None] * (self.commands[:, :2] - lin_xy) - self.cfg.root_lin_damping * lin_xy
        )
        self.root_state.root_lin_vel_b[:, :2] = lin_xy

        roll_acc = self.cfg.balance_gain * ((right_support - left_support) + 0.1 * (right_actions - left_actions))
        pitch_acc = self.cfg.balance_gain * ((rear_support - front_support) + 0.1 * (rear_actions - front_actions))
        yaw_acc = self.cfg.yaw_tracking_gain * (
            self.commands[:, 2] - self.root_state.root_ang_vel_b[:, 2]
        )

        self.root_state.root_ang_vel_b[:, 0] = self.root_state.root_ang_vel_b[:, 0] + dt * (
            roll_acc - self.cfg.root_ang_damping * self.root_state.root_ang_vel_b[:, 0]
        )
        self.root_state.root_ang_vel_b[:, 1] = self.root_state.root_ang_vel_b[:, 1] + dt * (
            pitch_acc - self.cfg.root_ang_damping * self.root_state.root_ang_vel_b[:, 1]
        )
        self.root_state.root_ang_vel_b[:, 2] = self.root_state.root_ang_vel_b[:, 2] + dt * (
            yaw_acc - self.cfg.root_ang_damping * self.root_state.root_ang_vel_b[:, 2]
        )

        extension = self._leg_extension(self.joint_state.joint_pos)
        target_height = (
            self.cfg.default_root_height
            + 0.15 * (mx.mean(extension, axis=1) - self.cfg.nominal_leg_extension)
            + 0.03 * (support_ratio - 1.0)
        )
        orientation_penalty = mx.sum(mx.square(self.root_state.projected_gravity_b[:, :2]), axis=1)
        z_acc = (
            self.cfg.height_stiffness * (target_height - self.root_state.root_pos_w[:, 2])
            - self.cfg.height_damping * self.root_state.root_lin_vel_b[:, 2]
            - self.cfg.orientation_height_penalty * orientation_penalty
        )
        self.root_state.root_lin_vel_b[:, 2] = self.root_state.root_lin_vel_b[:, 2] + dt * z_acc

        delta_quat = _quat_from_angular_velocity(self.root_state.root_ang_vel_b, dt)
        self.root_state.root_quat_w = _quat_normalize(_quat_multiply(self.root_state.root_quat_w, delta_quat))
        world_vel = _quat_rotate(self.root_state.root_quat_w, self.root_state.root_lin_vel_b)
        self.root_state.root_pos_w = self.root_state.root_pos_w + dt * world_vel

        gravity_w = mx.tile(mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32), (self.num_envs, 1))
        self.root_state.projected_gravity_b = _quat_rotate(_quat_conjugate(self.root_state.root_quat_w), gravity_w)

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        return self.joint_state.read()

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Any | None = None,
    ) -> None:
        del articulation
        self.joint_state.set_effort_target(efforts, joint_ids=joint_ids)

    def write_joint_state(
        self,
        articulation: Any,
        joint_pos: Any,
        joint_vel: Any,
        *,
        joint_acc: Any | None = None,
        env_ids: Any | None = None,
    ) -> None:
        del articulation, joint_acc
        self.joint_state.write(joint_pos, joint_vel, env_ids=env_ids)

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Any | None = None) -> None:
        del articulation
        self.root_state.write_root_pose(root_pose, env_ids=env_ids)

    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Any | None = None,
    ) -> None:
        del articulation
        self.root_state.write_root_velocity(root_velocity, env_ids=env_ids)

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "num_envs": self.num_envs,
            "capabilities": self.capabilities.__dict__,
            "contract": {
                "reset_signature": self.contract.reset_signature,
                "step_signature": self.contract.step_signature,
                "articulations": self.contract.articulations.__dict__,
            },
            "subsystems": {
                "terrain": True,
                "contacts": True,
                "deterministic_resets": True,
                "rollout_helpers": True,
            },
            "root_state_shape": list(self.root_state.root_pos_w.shape),
            "joint_state_shape": list(self.joint_state.joint_pos.shape),
            "terrain": self.terrain.state_dict(),
            "contacts": self.contact_model.state_dict(),
        }


class MacAnymalCFlatEnv:
    """A vectorized flat ANYmal-C locomotion environment on MLX/mac-sim."""

    def __init__(self, cfg: MacAnymalCFlatEnvCfg | None = None):
        self.cfg = cfg or MacAnymalCFlatEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)

        self.sim_backend = MacAnymalCFlatSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._episode_sums = {
            key: mx.zeros((self.num_envs,), dtype=mx.float32)
            for key in (
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            )
        }
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def reset(self) -> tuple[dict[str, mx.array], dict[str, Any]]:
        env_ids = list(range(self.num_envs))
        self._reset_idx(env_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, {}

    def step(self, actions: Any) -> tuple[dict[str, mx.array], mx.array, mx.array, mx.array, dict[str, Any]]:
        self._pre_physics_step(actions)
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.sim_backend.step(render=False)

        self.episode_length_buf = self.episode_length_buf + 1
        self.reward_buf = self._get_rewards()
        self.episode_return_buf = self.episode_return_buf + self.reward_buf
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        reset_ids = [index for index, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            extras = {
                "completed_lengths": [int(self.episode_length_buf[index].item()) for index in reset_ids],
                "completed_returns": [float(self.episode_return_buf[index].item()) for index in reset_ids],
            }
            self._reset_idx(reset_ids)

        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, extras

    def _pre_physics_step(self, actions: Any) -> None:
        self._actions = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space)), -1.0, 1.0)

    def _apply_action(self) -> None:
        self.sim_backend.set_action_targets(self._actions)

    def _get_observations(self) -> dict[str, mx.array]:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        rel_joint_pos = joint_pos - mx.broadcast_to(self.sim_backend.default_joint_pos, joint_pos.shape)
        obs = mx.concatenate(
            [
                self.sim_backend.root_lin_vel_b,
                self.sim_backend.root_ang_vel_b,
                self.sim_backend.projected_gravity_b,
                self.sim_backend.commands,
                rel_joint_pos,
                joint_vel,
                self._actions,
            ],
            axis=-1,
        )
        self._previous_actions = self._actions
        return {"policy": obs}

    def _get_rewards(self) -> mx.array:
        rewards = {
            "track_lin_vel_xy_exp": track_linear_velocity_xy_exp(
                self.sim_backend.commands, self.sim_backend.root_lin_vel_b
            )
            * self.cfg.lin_vel_reward_scale
            * self.step_dt,
            "track_ang_vel_z_exp": track_yaw_rate_z_exp(
                self.sim_backend.commands, self.sim_backend.root_ang_vel_b
            )
            * self.cfg.yaw_rate_reward_scale
            * self.step_dt,
            "lin_vel_z_l2": mx.square(self.sim_backend.root_lin_vel_b[:, 2]) * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": mx.sum(mx.square(self.sim_backend.root_ang_vel_b[:, :2]), axis=1)
            * self.cfg.ang_vel_reward_scale
            * self.step_dt,
            "dof_torques_l2": mx.sum(mx.square(self.sim_backend.applied_torque), axis=1)
            * self.cfg.joint_torque_reward_scale
            * self.step_dt,
            "dof_acc_l2": mx.sum(mx.square(self.sim_backend.joint_acc), axis=1)
            * self.cfg.joint_accel_reward_scale
            * self.step_dt,
            "action_rate_l2": action_rate_l2(self._actions, self._previous_actions)
            * self.cfg.action_rate_reward_scale
            * self.step_dt,
            "feet_air_time": feet_air_time_reward(self.sim_backend.contact_model, self.sim_backend.commands)
            * self.cfg.feet_air_time_reward_scale
            * self.step_dt,
            "undesired_contacts": undesired_contacts(
                self.sim_backend.contact_model,
                THIGH_BODY_NAMES,
                threshold=self.cfg.contact_force_threshold,
            )
            * self.cfg.undesired_contact_reward_scale
            * self.step_dt,
            "flat_orientation_l2": flat_orientation_l2(self.sim_backend.projected_gravity_b)
            * self.cfg.flat_orientation_reward_scale
            * self.step_dt,
        }
        for key, value in rewards.items():
            self._episode_sums[key] = self._episode_sums[key] + value
        return mx.sum(mx.stack(list(rewards.values())), axis=0)

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        base_contact = base_contact_termination(
            self.sim_backend.contact_model,
            BASE_BODY_NAMES,
            threshold=self.cfg.contact_force_threshold,
        )
        fell = self.sim_backend.root_pos_w[:, 2] < self.cfg.min_root_height
        return base_contact | fell, time_out

    def _reset_idx(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        self.sim_backend.reset_envs(env_ids)
        ids = mx.array(env_ids, dtype=mx.int32)
        self._actions[ids] = 0.0
        self._previous_actions[ids] = 0.0
        self.reward_buf[ids] = 0.0
        self.episode_return_buf[ids] = 0.0
        self.episode_length_buf[ids] = 0
        for value in self._episode_sums.values():
            value[ids] = 0.0


class MacAnymalCPolicy(nn.Module):
    """Continuous policy/value MLP for the mac-native ANYmal-C slice."""

    def __init__(self, obs_dim: int = 48, hidden_dim: int = 128, action_dim: int = 12):
        super().__init__()
        self.backbone = [
            nn.Linear(obs_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ]
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, obs: mx.array) -> tuple[mx.array, mx.array]:
        x = obs
        for layer in self.backbone:
            x = nn.elu(layer(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def _gaussian_log_probs(actions: mx.array, mean: mx.array, std: float) -> mx.array:
    variance = std * std
    return -0.5 * mx.sum(mx.square(actions - mean) / variance + math.log(variance) + LOG_2_PI, axis=-1)


def _gaussian_entropy(action_dim: int, std: float) -> float:
    return action_dim * (0.5 + 0.5 * LOG_2_PI + math.log(std))


def _ppo_loss(
    model: MacAnymalCPolicy,
    obs: mx.array,
    actions: mx.array,
    old_log_probs: mx.array,
    advantages: mx.array,
    returns: mx.array,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    action_std: float,
) -> mx.array:
    mean, values = model(obs)
    log_probs = _gaussian_log_probs(actions, mean, action_std)
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -mx.mean(mx.minimum(ratio * advantages, clipped_ratio * advantages))
    value_loss = 0.5 * mx.mean(mx.square(returns - values))
    entropy_bonus = _gaussian_entropy(actions.shape[1], action_std)
    return policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus


def train_anymal_c_policy(cfg: MacAnymalCTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control locomotion policy on the mac-native ANYmal-C slice."""

    mx.random.seed(cfg.env.seed)
    env = MacAnymalCFlatEnv(cfg.env)
    model = MacAnymalCPolicy(
        obs_dim=cfg.env.observation_space,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.env.action_space,
    )
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
    resumed_from = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from does not exist: {resume_path}")
        model.load_weights(str(resume_path))
        resumed_from = str(resume_path)
    mx.eval(model.parameters())

    obs = env.reset()[0]["policy"]
    completed_returns: list[float] = []

    for update in range(cfg.updates):
        obs_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        rewards_rollout = []
        dones_rollout = []
        values_rollout = []

        for _ in range(cfg.rollout_steps):
            mean, values = model(obs)
            noise = mx.random.normal(shape=mean.shape).astype(mx.float32)
            actions = mx.clip(mean + cfg.action_std * noise, -1.0, 1.0)
            next_obs, reward, terminated, truncated, extras = env.step(actions)
            done = terminated | truncated

            obs_rollout.append(obs)
            actions_rollout.append(actions)
            log_probs_rollout.append(_gaussian_log_probs(actions, mean, cfg.action_std))
            rewards_rollout.append(reward)
            dones_rollout.append(done.astype(mx.float32))
            values_rollout.append(values)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        _, bootstrap_value = model(obs)
        rewards_t = mx.stack(rewards_rollout)
        dones_t = mx.stack(dones_rollout)
        values_t = mx.stack(values_rollout)
        advantages = []
        gae = mx.zeros((cfg.env.num_envs,), dtype=mx.float32)

        for step in reversed(range(cfg.rollout_steps)):
            next_value = bootstrap_value if step == cfg.rollout_steps - 1 else values_t[step + 1]
            mask = 1.0 - dones_t[step]
            delta = rewards_t[step] + cfg.gamma * next_value * mask - values_t[step]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages.append(gae)

        advantages = mx.stack(list(reversed(advantages)))
        returns = advantages + values_t

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1, cfg.env.action_space))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))

        adv_mean = mx.mean(flat_advantages)
        adv_std = mx.sqrt(mx.mean(mx.square(flat_advantages - adv_mean)) + 1e-8)
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        for _ in range(cfg.epochs_per_update):
            loss, grads = loss_and_grad(
                model,
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_advantages,
                flat_returns,
                cfg.clip_epsilon,
                cfg.value_loss_coef,
                cfg.entropy_coef,
                cfg.action_std,
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = float(mx.mean(rewards_t).item())
            mean_return = sum(completed_returns[-10:]) / max(1, min(len(completed_returns), 10))
            print(
                f"[mlx-anymal-c-flat] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path = Path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path))
    metadata_path = _write_checkpoint_metadata(checkpoint_path, cfg)
    return {
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "resumed_from": resumed_from,
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": sum(completed_returns[-10:]) / max(1, min(len(completed_returns), 10)),
    }


def play_anymal_c_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacAnymalCFlatEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained ANYmal-C locomotion policy greedily and return episode returns."""

    cfg = env_cfg or MacAnymalCFlatEnvCfg(num_envs=1)
    env = MacAnymalCFlatEnv(cfg)
    checkpoint = Path(checkpoint_path)
    metadata = _read_checkpoint_metadata(checkpoint)
    policy_hidden_dim = hidden_dim or int(metadata.get("hidden_dim", 128))
    model = MacAnymalCPolicy(obs_dim=cfg.observation_space, hidden_dim=policy_hidden_dim, action_dim=cfg.action_space)
    model.load_weights(str(checkpoint))
    obs = env.reset()[0]["policy"]

    episode_returns: list[float] = []
    max_steps = max(1, env.max_episode_length * max(1, episodes) * 2)
    for _ in range(max_steps):
        mean, _ = model(obs)
        obs_dict, _, _, _, extras = env.step(mx.clip(mean, -1.0, 1.0))
        obs = obs_dict["policy"]
        if extras.get("completed_returns"):
            episode_returns.extend(float(value) for value in extras["completed_returns"])
        if len(episode_returns) >= episodes:
            break
    return episode_returns[:episodes]
