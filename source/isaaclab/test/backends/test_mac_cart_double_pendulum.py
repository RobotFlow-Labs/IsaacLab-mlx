# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for mac-native cart-double-pendulum MARL env."""

from __future__ import annotations

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import MacCartDoublePendulumEnv, MacCartDoublePendulumEnvCfg  # noqa: E402


def _random_actions(num_envs: int) -> dict[str, list[list[float]]]:
    return {
        "cart": [[0.5] for _ in range(num_envs)],
        "pendulum": [[-0.25] for _ in range(num_envs)],
    }


def test_mac_cart_double_pendulum_reset_and_step_shapes():
    """The MARL env should return per-agent observation/reward/done dictionaries."""
    cfg = MacCartDoublePendulumEnvCfg(num_envs=8, seed=11, episode_length_s=0.3)
    env = MacCartDoublePendulumEnv(cfg)

    obs, extras = env.reset()

    assert extras == {}
    assert set(obs.keys()) == {"cart", "pendulum"}
    assert obs["cart"].shape == (8, 4)
    assert obs["pendulum"].shape == (8, 3)

    next_obs, rewards, terminated, truncated, step_extras = env.step(_random_actions(cfg.num_envs))

    assert next_obs["cart"].shape == (8, 4)
    assert next_obs["pendulum"].shape == (8, 3)
    assert set(rewards.keys()) == {"cart", "pendulum"}
    assert rewards["cart"].shape == (8,)
    assert rewards["pendulum"].shape == (8,)
    assert set(terminated.keys()) == {"cart", "pendulum"}
    assert set(truncated.keys()) == {"cart", "pendulum"}
    assert terminated["cart"].shape == (8,)
    assert truncated["cart"].shape == (8,)
    assert isinstance(step_extras, dict)


def test_mac_cart_double_pendulum_termination_propagates_to_all_agents():
    """Out-of-bounds termination should mark both MARL agents and return reset extras."""
    cfg = MacCartDoublePendulumEnvCfg(num_envs=4, seed=5, episode_length_s=0.3)
    env = MacCartDoublePendulumEnv(cfg)

    joint_pos, joint_vel = env.sim_backend.joint_state()
    joint_pos = mx.array(joint_pos)
    joint_vel = mx.array(joint_vel)
    joint_pos[:, 0] = cfg.max_cart_pos + 1.0
    env.sim_backend.write_joint_state(None, joint_pos, joint_vel)

    _, _, terminated, _, extras = env.step(_random_actions(cfg.num_envs))

    assert bool(mx.all(terminated["cart"]).item())
    assert bool(mx.all(terminated["pendulum"]).item())
    assert "completed_returns" in extras
    assert set(extras["completed_returns"].keys()) == {"cart", "pendulum"}
