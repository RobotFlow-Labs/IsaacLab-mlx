# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native cart-double-pendulum environment with MARL-style dict interfaces."""

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

from .env_cfgs import MacCartDoublePendulumEnvCfg
from .reset_primitives import DeterministicResetSampler
from .state_primitives import BatchedArticulationState


def normalize_angle(angle: mx.array) -> mx.array:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_rewards(
    cfg: MacCartDoublePendulumEnvCfg,
    cart_pos: mx.array,
    cart_vel: mx.array,
    pole_pos: mx.array,
    pole_vel: mx.array,
    pendulum_pos: mx.array,
    pendulum_vel: mx.array,
    reset_terminated: mx.array,
) -> dict[str, mx.array]:
    terminated = reset_terminated.astype(mx.float32)
    rew_alive = cfg.rew_scale_alive * (1.0 - terminated)
    rew_termination = cfg.rew_scale_terminated * terminated
    rew_pole_pos = cfg.rew_scale_pole_pos * mx.square(pole_pos)
    rew_pendulum_pos = cfg.rew_scale_pendulum_pos * mx.square(pole_pos + pendulum_pos)
    rew_cart_vel = cfg.rew_scale_cart_vel * mx.abs(cart_vel)
    rew_pole_vel = cfg.rew_scale_pole_vel * mx.abs(pole_vel)
    rew_pendulum_vel = cfg.rew_scale_pendulum_vel * mx.abs(pendulum_vel)
    rew_cart_pos = cfg.rew_scale_cart_pos * mx.square(cart_pos)
    return {
        "cart": rew_alive + rew_termination + rew_cart_pos + rew_pole_pos + rew_cart_vel + rew_pole_vel,
        "pendulum": rew_alive + rew_termination + rew_pendulum_pos + rew_pendulum_vel,
    }


class MacCartDoublePendulumSimBackend(MacSimBackend):
    """A batched double-pendulum simulator with the same joint-level contract as Isaac Sim tasks."""

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

    def __init__(self, cfg: MacCartDoublePendulumEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.reset_sampler = reset_sampler or DeterministicResetSampler(cfg.seed)
        self.state = BatchedArticulationState(cfg.num_envs, num_joints=3)
        self.reset()

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    def joint_state(self) -> tuple[mx.array, mx.array]:
        return self.get_joint_state(None)

    def reset(self, *, soft: bool = False) -> None:
        self.reset_envs(list(range(self.num_envs)))

    def reset_envs(self, env_ids: list[int]) -> None:
        if len(env_ids) == 0:
            return
        joint_pos = mx.zeros((len(env_ids), 3), dtype=mx.float32)
        joint_vel = mx.zeros((len(env_ids), 3), dtype=mx.float32)
        joint_pos[:, 1] = self.reset_sampler.uniform(
            (len(env_ids),),
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
        )
        joint_pos[:, 2] = self.reset_sampler.uniform(
            (len(env_ids),),
            self.cfg.initial_pendulum_angle_range[0] * math.pi,
            self.cfg.initial_pendulum_angle_range[1] * math.pi,
        )
        self.state.reset_envs(env_ids, joint_pos=joint_pos, joint_vel=joint_vel, joint_effort_target=0.0)

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        joint_pos = self.state.joint_pos
        joint_vel = self.state.joint_vel
        cart_force = self.state.joint_effort_target[:, 0] * self.cfg.cart_force_scale
        pendulum_torque = self.state.joint_effort_target[:, 2] * self.cfg.pendulum_torque_scale

        total_mass = self.cfg.mass_cart + self.cfg.mass_pole + self.cfg.mass_pendulum
        costheta = mx.cos(joint_pos[:, 1])
        sintheta = mx.sin(joint_pos[:, 1])
        polemass_length = self.cfg.mass_pole * self.cfg.pole_half_length
        temp = (cart_force + polemass_length * mx.square(joint_vel[:, 1]) * sintheta) / total_mass
        pole_acc = (
            self.cfg.gravity * sintheta - costheta * temp
        ) / (
            self.cfg.pole_half_length
            * (4.0 / 3.0 - self.cfg.mass_pole * mx.square(costheta) / total_mass)
        )
        cart_acc = temp - polemass_length * pole_acc * costheta / total_mass

        pend_abs_angle = joint_pos[:, 1] + joint_pos[:, 2]
        inertia = self.cfg.mass_pendulum * self.cfg.pendulum_half_length * self.cfg.pendulum_half_length + 1e-6
        pendulum_acc = (
            -self.cfg.gravity * mx.sin(pend_abs_angle) / self.cfg.pendulum_half_length
            + pendulum_torque / inertia
            - self.cfg.pendulum_damping * joint_vel[:, 2]
            + 0.2 * pole_acc
        )

        dt = self.physics_dt
        self.state.joint_pos[:, 0] = joint_pos[:, 0] + dt * joint_vel[:, 0]
        self.state.joint_vel[:, 0] = joint_vel[:, 0] + dt * cart_acc
        self.state.joint_pos[:, 1] = joint_pos[:, 1] + dt * joint_vel[:, 1]
        self.state.joint_vel[:, 1] = joint_vel[:, 1] + dt * pole_acc
        self.state.joint_pos[:, 2] = joint_pos[:, 2] + dt * joint_vel[:, 2]
        self.state.joint_vel[:, 2] = joint_vel[:, 2] + dt * pendulum_acc

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        return self.state.read()

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Any | None = None,
    ) -> None:
        del articulation
        if joint_ids is None:
            joint_ids = [0]
        joint_ids = list(joint_ids)
        if any(joint_id not in (0, 2) for joint_id in joint_ids):
            raise ValueError(f"Unsupported joint_ids for cart-double-pendulum mac-sim: {joint_ids}")
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

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Any | None = None) -> None:
        del articulation, root_pose, env_ids

    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Any | None = None,
    ) -> None:
        del articulation, root_velocity, env_ids

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
                "terrain": False,
                "contacts": False,
                "deterministic_resets": True,
                "rollout_helpers": True,
            },
            "joint_state_shape": list(self.state.joint_pos.shape),
        }


class MacCartDoublePendulumEnv:
    """A vectorized MARL cart-double-pendulum env with per-agent dict interfaces."""

    def __init__(self, cfg: MacCartDoublePendulumEnvCfg | None = None):
        self.cfg = cfg or MacCartDoublePendulumEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.possible_agents = tuple(self.cfg.possible_agents)
        self.actions: dict[str, mx.array] = {
            "cart": mx.zeros((self.num_envs, 1), dtype=mx.float32),
            "pendulum": mx.zeros((self.num_envs, 1), dtype=mx.float32),
        }

        self.sim_backend = MacCartDoublePendulumSimBackend(
            self.cfg,
            reset_sampler=self.reset_sampler.fork("sim-backend"),
        )
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.episode_return_cart = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_pendulum = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.obs_buf = {
            "cart": mx.zeros((self.num_envs, 4), dtype=mx.float32),
            "pendulum": mx.zeros((self.num_envs, 3), dtype=mx.float32),
        }
        self.reset()

    def reset(self) -> tuple[dict[str, mx.array], dict[str, Any]]:
        env_ids = list(range(self.num_envs))
        self._reset_idx(env_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, {}

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, mx.array], dict[str, mx.array], dict[str, Any]]:
        self._pre_physics_step(actions)
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.sim_backend.step(render=False)

        self.episode_length_buf = self.episode_length_buf + 1
        joint_pos, joint_vel = self.sim_backend.joint_state()
        cart_pos = joint_pos[:, 0]
        cart_vel = joint_vel[:, 0]
        pole_pos = normalize_angle(joint_pos[:, 1])
        pole_vel = joint_vel[:, 1]
        pendulum_pos = normalize_angle(joint_pos[:, 2])
        pendulum_vel = joint_vel[:, 2]

        self.reset_terminated = (mx.abs(cart_pos) > self.cfg.max_cart_pos) | (mx.abs(pole_pos) > math.pi / 2.0)
        self.reset_time_outs = self.episode_length_buf >= (self.max_episode_length - 1)
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        rewards = compute_rewards(
            self.cfg,
            cart_pos,
            cart_vel,
            pole_pos,
            pole_vel,
            pendulum_pos,
            pendulum_vel,
            self.reset_terminated,
        )
        self.episode_return_cart = self.episode_return_cart + rewards["cart"]
        self.episode_return_pendulum = self.episode_return_pendulum + rewards["pendulum"]

        reset_ids = [idx for idx, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            extras = {
                "completed_returns": {
                    "cart": [float(self.episode_return_cart[idx].item()) for idx in reset_ids],
                    "pendulum": [float(self.episode_return_pendulum[idx].item()) for idx in reset_ids],
                },
                "completed_lengths": [int(self.episode_length_buf[idx].item()) for idx in reset_ids],
            }
            self._reset_idx(reset_ids)

        self.obs_buf = self._get_observations()
        terminated = {agent: self.reset_terminated for agent in self.possible_agents}
        truncated = {agent: self.reset_time_outs for agent in self.possible_agents}
        return self.obs_buf, rewards, terminated, truncated, extras

    def _pre_physics_step(self, actions: dict[str, Any]) -> None:
        cart = mx.array(actions["cart"], dtype=mx.float32).reshape((self.num_envs, 1))
        pendulum = mx.array(actions["pendulum"], dtype=mx.float32).reshape((self.num_envs, 1))
        self.actions["cart"] = mx.clip(cart, -1.0, 1.0)
        self.actions["pendulum"] = mx.clip(pendulum, -1.0, 1.0)

    def _apply_action(self) -> None:
        self.sim_backend.set_joint_effort_target(
            None, self.actions["cart"][:, 0] * self.cfg.cart_action_scale, joint_ids=[0]
        )
        self.sim_backend.set_joint_effort_target(
            None, self.actions["pendulum"][:, 0] * self.cfg.pendulum_action_scale, joint_ids=[2]
        )

    def _get_observations(self) -> dict[str, mx.array]:
        joint_pos, joint_vel = self.sim_backend.joint_state()
        pole_pos = normalize_angle(joint_pos[:, 1])
        pendulum_pos = normalize_angle(joint_pos[:, 2])
        return {
            "cart": mx.stack([joint_pos[:, 0], joint_vel[:, 0], pole_pos, joint_vel[:, 1]], axis=-1),
            "pendulum": mx.stack([pole_pos + pendulum_pos, pendulum_pos, joint_vel[:, 2]], axis=-1),
        }

    def _reset_idx(self, env_ids: list[int]) -> None:
        self.sim_backend.reset_envs(env_ids)
        ids = mx.array(env_ids)
        self.episode_length_buf[ids] = 0
        self.episode_return_cart[ids] = 0.0
        self.episode_return_pendulum[ids] = 0.0
