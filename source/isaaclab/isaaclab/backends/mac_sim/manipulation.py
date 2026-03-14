# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native Franka manipulation slices for the MLX/mac-sim path."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from isaaclab.backends.runtime import (
    ArticulationCapabilities,
    MacSimBackend,
    SimBackendContract,
    SimCapabilities,
    resolve_runtime_selection,
    set_runtime_selection,
)

from .env_cfgs import MacFrankaLiftEnvCfg, MacFrankaReachEnvCfg
from .reset_primitives import DeterministicResetSampler
from .state_primitives import BatchedArticulationState


def _franka_end_effector_position(joint_pos: mx.array) -> mx.array:
    """Approximate Franka end-effector pose with a deterministic analytic kinematic map."""

    q0 = joint_pos[:, 0]
    q1 = joint_pos[:, 1]
    q2 = joint_pos[:, 2]
    q3 = joint_pos[:, 3]
    q4 = joint_pos[:, 4]
    q5 = joint_pos[:, 5]
    q6 = joint_pos[:, 6]
    x = (
        0.28
        + 0.18 * mx.cos(q0) * mx.cos(q1)
        + 0.14 * mx.cos(q0) * mx.cos(q1 + q2)
        + 0.08 * mx.cos(q0) * mx.cos(q1 + q2 + 0.5 * q3)
        - 0.02 * mx.sin(q4)
    )
    y = 0.20 * mx.sin(q0) + 0.07 * mx.sin(q0 + 0.5 * q3) + 0.03 * mx.tanh(q5) + 0.015 * mx.sin(q6)
    z = 0.28 + 0.18 * mx.sin(-q1) + 0.13 * mx.sin(-(q1 + q2)) + 0.07 * mx.sin(-(q1 + q2 + 0.5 * q3))
    return mx.stack((x, y, z), axis=-1).astype(mx.float32)


class MacFrankaReachSimBackend(MacSimBackend):
    """A lightweight batched Franka reach backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=False,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )
    contract = SimBackendContract(
        reset_signature="reset(soft: bool = False) -> None",
        step_signature="step(render: bool = True, update_fabric: bool = False) -> None",
        articulations=ArticulationCapabilities(
            joint_state_io=True,
            root_state_io=False,
            effort_targets=True,
            batched_views=True,
        ),
    )

    def __init__(self, cfg: MacFrankaReachEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.reset_sampler = reset_sampler or DeterministicResetSampler(cfg.seed)
        self.state = BatchedArticulationState(cfg.num_envs, num_joints=cfg.action_space)
        self.default_joint_pos = mx.array(cfg.default_joint_pos, dtype=mx.float32).reshape((1, cfg.action_space))
        self.joint_lower_limits = mx.array(cfg.joint_lower_limits, dtype=mx.float32).reshape((1, cfg.action_space))
        self.joint_upper_limits = mx.array(cfg.joint_upper_limits, dtype=mx.float32).reshape((1, cfg.action_space))
        self.action_targets = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.joint_acc = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.applied_torque = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.target_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.ee_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.reset()

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    def reset(self, *, soft: bool = False) -> None:
        del soft
        self.reset_envs(list(range(self.num_envs)))

    def reset_envs(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        joint_pos = mx.broadcast_to(self.default_joint_pos, (rows, self.cfg.action_space)) + self.reset_sampler.uniform(
            (rows, self.cfg.action_space),
            -self.cfg.joint_reset_noise,
            self.cfg.joint_reset_noise,
        )
        joint_pos = mx.clip(
            joint_pos,
            mx.broadcast_to(self.joint_lower_limits, (rows, self.cfg.action_space)),
            mx.broadcast_to(self.joint_upper_limits, (rows, self.cfg.action_space)),
        )
        self.state.reset_envs(env_ids, joint_pos=joint_pos, joint_vel=0.0, joint_effort_target=0.0)
        self.action_targets[ids] = 0.0
        self.joint_acc[ids] = 0.0
        self.applied_torque[ids] = 0.0
        self.target_pos_w[ids, 0] = self.reset_sampler.uniform((rows,), self.cfg.target_x_range[0], self.cfg.target_x_range[1])
        self.target_pos_w[ids, 1] = self.reset_sampler.uniform((rows,), self.cfg.target_y_range[0], self.cfg.target_y_range[1])
        self.target_pos_w[ids, 2] = self.reset_sampler.uniform((rows,), self.cfg.target_z_range[0], self.cfg.target_z_range[1])
        self.ee_pos_w[ids] = _franka_end_effector_position(self.state.joint_pos[ids, :7])

    def set_action_targets(self, actions: Any) -> None:
        self.action_targets = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space)), -1.0, 1.0)

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        dt = self.physics_dt
        desired_joint_pos = self.default_joint_pos + self.cfg.action_scale * self.action_targets
        desired_joint_pos = mx.clip(
            desired_joint_pos,
            mx.broadcast_to(self.joint_lower_limits, desired_joint_pos.shape),
            mx.broadcast_to(self.joint_upper_limits, desired_joint_pos.shape),
        )
        pos_error = desired_joint_pos - self.state.joint_pos
        acc = self.cfg.joint_stiffness * pos_error - self.cfg.joint_damping * self.state.joint_vel
        acc = acc / self.cfg.joint_inertia
        self.state.joint_vel = self.state.joint_vel + dt * acc
        self.state.joint_pos = mx.clip(
            self.state.joint_pos + dt * self.state.joint_vel,
            mx.broadcast_to(self.joint_lower_limits, self.state.joint_pos.shape),
            mx.broadcast_to(self.joint_upper_limits, self.state.joint_pos.shape),
        )
        self.joint_acc = acc
        self.applied_torque = self.cfg.joint_stiffness * pos_error
        self.ee_pos_w = _franka_end_effector_position(self.state.joint_pos[:, :7])

    def goal_distance(self) -> mx.array:
        return mx.linalg.norm(self.target_pos_w - self.ee_pos_w, axis=1)

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        return self.state.read()

    def set_joint_effort_target(self, articulation: Any, efforts: Any, *, joint_ids: Any | None = None) -> None:
        del articulation
        self.state.set_effort_target(efforts, joint_ids=joint_ids)

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
        self.state.write(joint_pos, joint_vel, env_ids=env_ids)
        self.ee_pos_w = _franka_end_effector_position(self.state.joint_pos[:, :7])

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Any | None = None) -> None:
        del articulation, root_pose, env_ids

    def write_root_velocity(self, articulation: Any, root_velocity: Any, *, env_ids: Any | None = None) -> None:
        del articulation, root_velocity, env_ids

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "task": "franka-reach",
            "num_envs": self.num_envs,
            "capabilities": self.capabilities.__dict__,
            "contract": {
                "reset_signature": self.contract.reset_signature,
                "step_signature": self.contract.step_signature,
                "articulations": self.contract.articulations.__dict__,
            },
            "subsystems": {
                "analytic_kinematics": True,
                "object_tracking": False,
                "grasp_logic": False,
            },
            "joint_state_shape": list(self.state.joint_pos.shape),
        }


class MacFrankaLiftSimBackend(MacFrankaReachSimBackend):
    """A lightweight batched Franka cube-lift backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )

    def __init__(self, cfg: MacFrankaLiftEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.grasped = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        super().__init__(cfg, reset_sampler=reset_sampler)
        self.cfg = cfg

    def reset_envs(self, env_ids: list[int]) -> None:
        super().reset_envs(env_ids)
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        self.cube_pos_w[ids, 0] = self.reset_sampler.uniform((rows,), self.cfg.cube_x_range[0], self.cfg.cube_x_range[1])
        self.cube_pos_w[ids, 1] = self.reset_sampler.uniform((rows,), self.cfg.cube_y_range[0], self.cfg.cube_y_range[1])
        self.cube_pos_w[ids, 2] = self.cfg.table_height
        self.grasped[ids] = False

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        super().step(render=render, update_fabric=update_fabric)
        gripper_target = mx.where(
            self.action_targets[:, 7] < 0.0,
            self.joint_lower_limits[0, 7],
            self.joint_upper_limits[0, 7],
        ).astype(mx.float32)
        self.state.joint_vel[:, 7] = (gripper_target - self.state.joint_pos[:, 7]) / self.physics_dt
        self.state.joint_pos[:, 7] = gripper_target
        ee_to_cube = self.cube_pos_w - self.ee_pos_w
        dist_to_cube = mx.linalg.norm(ee_to_cube, axis=1)
        gripper_closed = self.state.joint_pos[:, 7] <= self.cfg.gripper_closed_threshold
        can_grasp = dist_to_cube <= self.cfg.grasp_distance_threshold
        self.grasped = (self.grasped | (can_grasp & gripper_closed)) & gripper_closed
        attached_cube = self.ee_pos_w + mx.array([0.0, 0.0, -self.cfg.grasp_offset_z], dtype=mx.float32)
        resting_cube = self.cube_pos_w
        resting_cube[:, 2] = mx.maximum(self.cfg.table_height, resting_cube[:, 2] - self.physics_dt * 0.35)
        self.cube_pos_w = mx.where(self.grasped[:, None], attached_cube, resting_cube)

    def lift_success(self) -> mx.array:
        return self.cube_pos_w[:, 2] >= self.cfg.lift_success_height

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-lift"
        payload["subsystems"] = {
            "analytic_kinematics": True,
            "object_tracking": True,
            "grasp_logic": True,
        }
        return payload


class MacFrankaReachEnv:
    """Vectorized Franka reach task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaReachEnvCfg | None = None):
        self.cfg = cfg or MacFrankaReachEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaReachSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def reset(self) -> tuple[dict[str, mx.array], dict[str, Any]]:
        self._reset_idx(list(range(self.num_envs)))
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
        step_reward = self.reward_buf
        step_terminated = self.reset_terminated
        step_time_outs = self.reset_time_outs

        reset_ids = [index for index, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            ids = mx.array(reset_ids, dtype=mx.int32)
            extras = {
                "completed_lengths": [int(self.episode_length_buf[index].item()) for index in reset_ids],
                "completed_returns": [float(self.episode_return_buf[index].item()) for index in reset_ids],
                "terminated_env_ids": [index for index in reset_ids if bool(step_terminated[index].item())],
                "truncated_env_ids": [index for index in reset_ids if bool(step_time_outs[index].item())],
                "final_policy_observations": self._build_policy_observations()[ids],
            }
            self._reset_idx(reset_ids)

        self.obs_buf = self._get_observations()
        return self.obs_buf, step_reward, step_terminated, step_time_outs, extras

    def _pre_physics_step(self, actions: Any) -> None:
        self._actions = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space)), -1.0, 1.0)

    def _apply_action(self) -> None:
        self.sim_backend.set_action_targets(self._actions)

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        ee_error = self.sim_backend.target_pos_w - self.sim_backend.ee_pos_w
        return mx.concatenate((joint_pos, joint_vel, self.sim_backend.ee_pos_w, self.sim_backend.target_pos_w, ee_error), axis=-1)

    def _get_observations(self) -> dict[str, mx.array]:
        obs = self._build_policy_observations()
        self._previous_actions = self._actions
        return {"policy": obs}

    def _get_rewards(self) -> mx.array:
        distance = self.sim_backend.goal_distance()
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * distance)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        joint_vel_penalty = self.cfg.joint_vel_penalty_scale * mx.sum(mx.square(self.sim_backend.state.joint_vel), axis=1)
        success_bonus = self.cfg.success_bonus * (distance <= self.cfg.success_threshold).astype(mx.float32)
        return (reach_reward + action_penalty + joint_vel_penalty + success_bonus) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.goal_distance() <= self.cfg.success_threshold
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out

    def _reset_idx(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        self.sim_backend.reset_envs(env_ids)
        ids = mx.array(env_ids, dtype=mx.int32)
        self._actions[ids] = 0.0
        self._previous_actions[ids] = 0.0
        self.episode_return_buf[ids] = 0.0
        self.episode_length_buf[ids] = 0


class MacFrankaLiftEnv(MacFrankaReachEnv):
    """Vectorized Franka cube-lift task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaLiftEnvCfg | None = None):
        self.cfg = cfg or MacFrankaLiftEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaLiftSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        cube_error = self.sim_backend.cube_pos_w - self.sim_backend.ee_pos_w
        goal_gap = (self.cfg.lift_success_height - self.sim_backend.cube_pos_w[:, 2]).reshape((-1, 1))
        grasped = self.sim_backend.grasped.astype(mx.float32).reshape((-1, 1))
        return mx.concatenate(
            (joint_pos, joint_vel, self.sim_backend.ee_pos_w, self.sim_backend.cube_pos_w, cube_error, goal_gap, grasped),
            axis=-1,
        )

    def _get_rewards(self) -> mx.array:
        cube_error = mx.linalg.norm(self.sim_backend.cube_pos_w - self.sim_backend.ee_pos_w, axis=1)
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * cube_error)
        grasp_reward = self.cfg.grasp_reward_scale * self.sim_backend.grasped.astype(mx.float32)
        lift_reward = self.cfg.lift_reward_scale * mx.maximum(self.sim_backend.cube_pos_w[:, 2] - self.cfg.table_height, 0.0)
        success_bonus = self.cfg.lift_success_bonus * self.sim_backend.lift_success().astype(mx.float32)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        return (reach_reward + grasp_reward + lift_reward + success_bonus + action_penalty) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.lift_success()
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out
