# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native quadcopter environment with root-state dynamics."""

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
from isaaclab.utils.configclass import configclass


@configclass
class MacQuadcopterEnvCfg:
    """Configuration aligned with the upstream quadcopter task where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 100.0
    decimation: int = 2
    episode_length_s: float = 10.0
    action_space: int = 4
    observation_space: int = 12
    state_space: int = 0

    env_spacing: float = 2.5
    mass: float = 0.032
    gravity: float = 9.81
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    angular_damping: float = 0.08
    linear_damping_xy: float = 0.05
    linear_damping_z: float = 0.03
    lateral_accel_scale: float = 2.5

    min_height: float = 0.1
    max_height: float = 2.0

    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0

    seed: int = 42


def _grid_origins(num_envs: int, spacing: float) -> mx.array:
    side = int(math.ceil(math.sqrt(num_envs)))
    grid_x = mx.arange(side, dtype=mx.float32)
    grid_y = mx.arange(side, dtype=mx.float32)
    mesh_x = mx.repeat(grid_x.reshape((1, -1)), side, axis=0).reshape((-1,))
    mesh_y = mx.repeat(grid_y.reshape((-1, 1)), side, axis=1).reshape((-1,))
    centered_x = (mesh_x[:num_envs] - (side - 1) / 2.0) * spacing
    centered_y = (mesh_y[:num_envs] - (side - 1) / 2.0) * spacing
    zeros = mx.zeros((num_envs,), dtype=mx.float32)
    return mx.stack([centered_x, centered_y, zeros], axis=-1)


def _quat_conjugate(quat: mx.array) -> mx.array:
    return mx.concatenate([-quat[:, :3], quat[:, 3:4]], axis=-1)


def _quat_normalize(quat: mx.array, eps: float = 1e-8) -> mx.array:
    norm = mx.linalg.norm(quat, axis=1, keepdims=True)
    return quat / mx.maximum(norm, eps)


def _quat_multiply(lhs: mx.array, rhs: mx.array) -> mx.array:
    x1, y1, z1, w1 = lhs[:, 0], lhs[:, 1], lhs[:, 2], lhs[:, 3]
    x2, y2, z2, w2 = rhs[:, 0], rhs[:, 1], rhs[:, 2], rhs[:, 3]
    return mx.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(quat: mx.array, vector: mx.array) -> mx.array:
    quat_xyz = quat[:, :3]
    quat_w = quat[:, 3:4]
    cross_1 = mx.linalg.cross(quat_xyz, vector, axis=1)
    cross_2 = mx.linalg.cross(quat_xyz, cross_1, axis=1)
    return vector + 2.0 * (quat_w * cross_1 + cross_2)


def _quat_from_angular_velocity(omega_body: mx.array, dt: float) -> mx.array:
    delta = omega_body * dt
    angle = mx.linalg.norm(delta, axis=1, keepdims=True)
    half_angle = 0.5 * angle
    axis = delta / mx.maximum(angle, 1e-8)
    sin_half = mx.sin(half_angle)
    cos_half = mx.cos(half_angle)
    return _quat_normalize(mx.concatenate([axis * sin_half, cos_half], axis=-1))


class MacQuadcopterSimBackend(MacSimBackend):
    """A lightweight root-state quadcopter simulator for MLX/mac-sim."""

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
            root_state_io=True,
            effort_targets=True,
            batched_views=True,
        ),
    )

    def __init__(self, cfg: MacQuadcopterEnvCfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.env_origins = _grid_origins(cfg.num_envs, cfg.env_spacing)
        self.root_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.root_lin_vel_b = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.root_ang_vel_b = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.root_quat_w = mx.tile(mx.array([[0.0, 0.0, 0.0, 1.0]], dtype=mx.float32), (cfg.num_envs, 1))
        self.projected_gravity_b = mx.tile(mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32), (cfg.num_envs, 1))
        self._thrust = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self._moments = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.reset()

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    def reset(self, *, soft: bool = False) -> None:
        self.reset_envs(list(range(self.num_envs)))

    def reset_envs(self, env_ids: list[int]) -> None:
        if len(env_ids) == 0:
            return
        ids = mx.array(env_ids)
        self.root_pos_w[ids] = self.env_origins[ids] + mx.array([0.0, 0.0, 1.0], dtype=mx.float32)
        self.root_lin_vel_b[ids] = 0.0
        self.root_ang_vel_b[ids] = 0.0
        self.root_quat_w[ids] = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)
        self.projected_gravity_b[ids] = mx.array([0.0, 0.0, -1.0], dtype=mx.float32)
        self._thrust[ids] = self.cfg.mass * self.cfg.gravity
        self._moments[ids] = 0.0

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        dt = self.physics_dt
        thrust_accel_z = self._thrust / self.cfg.mass - self.cfg.gravity
        lateral_accel = self.cfg.lateral_accel_scale * self._moments[:, :2]

        accel = mx.stack([lateral_accel[:, 0], lateral_accel[:, 1], thrust_accel_z], axis=-1)
        damping = mx.array(
            [self.cfg.linear_damping_xy, self.cfg.linear_damping_xy, self.cfg.linear_damping_z], dtype=mx.float32
        )
        self.root_lin_vel_b = self.root_lin_vel_b + dt * (accel - damping * self.root_lin_vel_b)
        self.root_pos_w = self.root_pos_w + dt * self.root_lin_vel_b

        ang_acc = self._moments - self.cfg.angular_damping * self.root_ang_vel_b
        self.root_ang_vel_b = self.root_ang_vel_b + dt * ang_acc
        delta_quat = _quat_from_angular_velocity(self.root_ang_vel_b, dt)
        self.root_quat_w = _quat_normalize(_quat_multiply(self.root_quat_w, delta_quat))
        gravity_w = mx.tile(mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32), (self.num_envs, 1))
        self.projected_gravity_b = _quat_rotate(_quat_conjugate(self.root_quat_w), gravity_w)

    def set_thrust_and_moment(self, thrust: Any, moment: Any) -> None:
        self._thrust = mx.array(thrust, dtype=mx.float32).reshape((self.num_envs,))
        self._moments = mx.array(moment, dtype=mx.float32).reshape((self.num_envs, 3))

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        joint_pos = mx.zeros((self.num_envs, 4), dtype=mx.float32)
        joint_vel = mx.zeros((self.num_envs, 4), dtype=mx.float32)
        return joint_pos, joint_vel

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Any | None = None,
    ) -> None:
        del articulation, efforts, joint_ids

    def write_joint_state(
        self,
        articulation: Any,
        joint_pos: Any,
        joint_vel: Any,
        *,
        joint_acc: Any | None = None,
        env_ids: Any | None = None,
    ) -> None:
        del articulation, joint_pos, joint_vel, joint_acc, env_ids

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Any | None = None) -> None:
        del articulation
        pose = mx.array(root_pose, dtype=mx.float32)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        ids = mx.array(env_ids)
        self.root_pos_w[ids] = pose[:, :3]
        if pose.shape[1] >= 7:
            self.root_quat_w[ids] = pose[:, 3:7]

    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Any | None = None,
    ) -> None:
        del articulation
        velocity = mx.array(root_velocity, dtype=mx.float32)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        ids = mx.array(env_ids)
        self.root_lin_vel_b[ids] = velocity[:, :3]
        if velocity.shape[1] >= 6:
            self.root_ang_vel_b[ids] = velocity[:, 3:6]

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "num_envs": self.num_envs,
            "root_pos_w": self.root_pos_w.tolist(),
            "root_lin_vel_b": self.root_lin_vel_b.tolist(),
            "root_ang_vel_b": self.root_ang_vel_b.tolist(),
        }


class MacQuadcopterEnv:
    """A vectorized quadcopter env that mirrors upstream DirectRL task structure."""

    def __init__(self, cfg: MacQuadcopterEnvCfg | None = None):
        self.cfg = cfg or MacQuadcopterEnvCfg()
        mx.random.seed(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)

        self.sim_backend = MacQuadcopterSimBackend(self.cfg)
        self._actions = mx.zeros((self.num_envs, 4), dtype=mx.float32)
        self._thrust = mx.zeros((self.num_envs,), dtype=mx.float32)
        self._moment = mx.zeros((self.num_envs, 3), dtype=mx.float32)
        self._desired_pos_w = mx.zeros((self.num_envs, 3), dtype=mx.float32)

        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._episode_sums = {
            "lin_vel": mx.zeros((self.num_envs,), dtype=mx.float32),
            "ang_vel": mx.zeros((self.num_envs,), dtype=mx.float32),
            "distance_to_goal": mx.zeros((self.num_envs,), dtype=mx.float32),
        }
        self.obs_buf = {"policy": mx.zeros((self.num_envs, 12), dtype=mx.float32)}
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
        rewards = self._get_rewards()
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        reset_ids = [idx for idx, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            final_distance = mx.linalg.norm(
                self._desired_pos_w[mx.array(reset_ids)] - self.sim_backend.root_pos_w[mx.array(reset_ids)], axis=1
            )
            extras = {
                "completed_lengths": [int(self.episode_length_buf[idx].item()) for idx in reset_ids],
                "final_distance_to_goal": float(mx.mean(final_distance).item()),
            }
            self._reset_idx(reset_ids)

        self.obs_buf = self._get_observations()
        return self.obs_buf, rewards, self.reset_terminated, self.reset_time_outs, extras

    def _pre_physics_step(self, actions: Any) -> None:
        self._actions = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, 4)), -1.0, 1.0)
        hover_thrust = self.cfg.thrust_to_weight * self.cfg.mass * self.cfg.gravity
        self._thrust = hover_thrust * (self._actions[:, 0] + 1.0) / 2.0
        self._moment = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
        self.sim_backend.set_thrust_and_moment(self._thrust, self._moment)

    def _get_observations(self) -> dict[str, mx.array]:
        desired_pos_w = self._desired_pos_w - self.sim_backend.root_pos_w
        desired_pos_b = _quat_rotate(_quat_conjugate(self.sim_backend.root_quat_w), desired_pos_w)
        obs = mx.concatenate(
            [
                self.sim_backend.root_lin_vel_b,
                self.sim_backend.root_ang_vel_b,
                self.sim_backend.projected_gravity_b,
                desired_pos_b,
            ],
            axis=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> mx.array:
        lin_vel = mx.sum(mx.square(self.sim_backend.root_lin_vel_b), axis=1)
        ang_vel = mx.sum(mx.square(self.sim_backend.root_ang_vel_b), axis=1)
        distance_to_goal = mx.linalg.norm(self._desired_pos_w - self.sim_backend.root_pos_w, axis=1)
        distance_to_goal_mapped = 1.0 - mx.tanh(distance_to_goal / 0.8)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        for key, value in rewards.items():
            self._episode_sums[key] = self._episode_sums[key] + value
        return rewards["lin_vel"] + rewards["ang_vel"] + rewards["distance_to_goal"]

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        died = (
            (self.sim_backend.root_pos_w[:, 2] < self.cfg.min_height)
            | (self.sim_backend.root_pos_w[:, 2] > self.cfg.max_height)
        )
        return died, time_out

    def _reset_idx(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        self.sim_backend.reset_envs(env_ids)
        ids = mx.array(env_ids)

        self._actions[ids] = 0.0
        self.episode_length_buf[ids] = 0
        for key in self._episode_sums:
            self._episode_sums[key][ids] = 0.0

        desired = mx.zeros((len(env_ids), 3), dtype=mx.float32)
        desired[:, :2] = mx.random.uniform(low=-2.0, high=2.0, shape=(len(env_ids), 2))
        desired[:, :2] = desired[:, :2] + self.sim_backend.env_origins[ids, :2]
        desired[:, 2] = mx.random.uniform(low=0.5, high=1.5, shape=(len(env_ids),))
        self._desired_pos_w[ids] = desired
