# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole showcase environments for exercising non-Box action and observation spaces on macOS."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mlx.core as mx

from isaaclab.backends.mac_sim.cartpole import MacCartpoleEnv, MacCartpoleEnvCfg
from isaaclab.utils.configclass import configclass


def _box_obs() -> spaces.Box:
    return spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))


def _joint_box() -> spaces.Box:
    return spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,))


def _box_action() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(1,))


def _discrete_obs() -> spaces.Discrete:
    return spaces.Discrete(16)


def _discrete_action() -> spaces.Discrete:
    return spaces.Discrete(3)


def _multidiscrete_obs() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([2, 2, 2, 2])


def _multidiscrete_action() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([3, 2])


def _dict_obs() -> spaces.Dict:
    return spaces.Dict({"joint-positions": _joint_box(), "joint-velocities": _joint_box()})


def _tuple_obs() -> spaces.Tuple:
    return spaces.Tuple((_joint_box(), _joint_box()))


@configclass
class BoxBoxEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _box_obs()
    action_space: Any = _box_action()


@configclass
class BoxDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _box_obs()
    action_space: Any = _discrete_action()


@configclass
class BoxMultiDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _box_obs()
    action_space: Any = _multidiscrete_action()


@configclass
class DiscreteBoxEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _discrete_obs()
    action_space: Any = _box_action()


@configclass
class DiscreteDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _discrete_obs()
    action_space: Any = _discrete_action()


@configclass
class DiscreteMultiDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _discrete_obs()
    action_space: Any = _multidiscrete_action()


@configclass
class MultiDiscreteBoxEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _multidiscrete_obs()
    action_space: Any = _box_action()


@configclass
class MultiDiscreteDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _multidiscrete_obs()
    action_space: Any = _discrete_action()


@configclass
class MultiDiscreteMultiDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _multidiscrete_obs()
    action_space: Any = _multidiscrete_action()


@configclass
class DictBoxEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _dict_obs()
    action_space: Any = _box_action()


@configclass
class DictDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _dict_obs()
    action_space: Any = _discrete_action()


@configclass
class DictMultiDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _dict_obs()
    action_space: Any = _multidiscrete_action()


@configclass
class TupleBoxEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _tuple_obs()
    action_space: Any = _box_action()


@configclass
class TupleDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _tuple_obs()
    action_space: Any = _discrete_action()


@configclass
class TupleMultiDiscreteEnvCfg(MacCartpoleEnvCfg):
    observation_space: Any = _tuple_obs()
    action_space: Any = _multidiscrete_action()


SHOWCASE_CFGS = (
    BoxBoxEnvCfg,
    BoxDiscreteEnvCfg,
    BoxMultiDiscreteEnvCfg,
    DiscreteBoxEnvCfg,
    DiscreteDiscreteEnvCfg,
    DiscreteMultiDiscreteEnvCfg,
    MultiDiscreteBoxEnvCfg,
    MultiDiscreteDiscreteEnvCfg,
    MultiDiscreteMultiDiscreteEnvCfg,
    DictBoxEnvCfg,
    DictDiscreteEnvCfg,
    DictMultiDiscreteEnvCfg,
    TupleBoxEnvCfg,
    TupleDiscreteEnvCfg,
    TupleMultiDiscreteEnvCfg,
)


def _sign_bits(joint_pos: mx.array, joint_vel: mx.array) -> mx.array:
    return mx.stack(
        [
            (joint_pos[:, 1] >= 0).astype(mx.int32),
            (joint_pos[:, 0] >= 0).astype(mx.int32),
            (joint_vel[:, 1] >= 0).astype(mx.int32),
            (joint_vel[:, 0] >= 0).astype(mx.int32),
        ],
        axis=-1,
    )


class MacCartpoleShowcaseEnv(MacCartpoleEnv):
    """Mac-native cartpole showcase mirroring the upstream action/observation space variations."""

    cfg: MacCartpoleEnvCfg

    def _pre_physics_step(self, action: Any) -> None:
        if isinstance(self.single_action_space, gym.spaces.Box):
            actions = mx.array(action, dtype=mx.float32).reshape((self.num_envs, 1))
            self.actions = mx.clip(actions, -1.0, 1.0)
            return
        if isinstance(self.single_action_space, gym.spaces.Discrete):
            self.actions = mx.array([int(item) for item in action], dtype=mx.int32).reshape((self.num_envs,))
            return
        if isinstance(self.single_action_space, gym.spaces.MultiDiscrete):
            normalized = [[int(value) for value in item] for item in action]
            self.actions = mx.array(normalized, dtype=mx.int32).reshape((self.num_envs, 2))
            return
        raise NotImplementedError(f"Action space {type(self.single_action_space)} not implemented")

    def _apply_action(self) -> None:
        if isinstance(self.single_action_space, gym.spaces.Box):
            target = self.cfg.action_scale * self.actions[:, 0]
        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            target = mx.zeros((self.num_envs,), dtype=mx.float32)
            target = mx.where(self.actions == 1, -self.cfg.action_scale, target)
            target = mx.where(self.actions == 2, self.cfg.action_scale, target)
        elif isinstance(self.single_action_space, gym.spaces.MultiDiscrete):
            magnitude = mx.zeros((self.num_envs,), dtype=mx.float32)
            magnitude = mx.where(self.actions[:, 0] == 1, self.cfg.action_scale / 2.0, magnitude)
            magnitude = mx.where(self.actions[:, 0] == 2, self.cfg.action_scale, magnitude)
            target = mx.where(self.actions[:, 1] == 0, -magnitude, magnitude)
        else:
            raise NotImplementedError(f"Action space {type(self.single_action_space)} not implemented")

        self.sim_backend.set_joint_effort_target(None, target, joint_ids=[0])

    def _get_observations(self) -> dict[str, Any]:
        joint_pos, joint_vel = self._joint_state()
        policy_space = self.single_observation_space["policy"]

        if isinstance(policy_space, gym.spaces.Box):
            obs = mx.stack([joint_pos[:, 1], joint_vel[:, 1], joint_pos[:, 0], joint_vel[:, 0]], axis=-1)
        elif isinstance(policy_space, gym.spaces.Discrete):
            bits = _sign_bits(joint_pos, joint_vel)
            obs = bits[:, 0] * 8 + bits[:, 1] * 4 + bits[:, 2] * 2 + bits[:, 3]
        elif isinstance(policy_space, gym.spaces.MultiDiscrete):
            obs = _sign_bits(joint_pos, joint_vel)
        elif isinstance(policy_space, gym.spaces.Tuple):
            obs = (joint_pos, joint_vel)
        elif isinstance(policy_space, gym.spaces.Dict):
            obs = {"joint-positions": joint_pos, "joint-velocities": joint_vel}
        else:
            raise NotImplementedError(f"Observation space {type(policy_space)} not implemented")

        return {"policy": obs}
