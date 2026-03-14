# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native Franka lift slice."""

from __future__ import annotations

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import MacFrankaLiftEnv, MacFrankaLiftEnvCfg  # noqa: E402


def test_mac_franka_lift_reset_and_step_shapes():
    """The Franka lift env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaLiftEnvCfg(num_envs=8, seed=31, episode_length_s=0.5)
    env = MacFrankaLiftEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 27)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 27)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    assert env.sim_backend.state_dict()["task"] == "franka-lift"


def test_mac_franka_lift_can_grasp_and_raise_cube():
    """Closing the gripper near the cube should engage the grasp logic and move the cube upward."""

    cfg = MacFrankaLiftEnvCfg(num_envs=2, seed=37, episode_length_s=0.5, lift_success_height=1.0)
    env = MacFrankaLiftEnv(cfg)
    env.sim_backend.cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array([0.0, 0.0, -0.02], dtype=mx.float32)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = -1.0
    _, _, _, _, _ = env.step(actions)

    assert bool(mx.any(env.sim_backend.grasped).item())
    assert float(mx.max(env.sim_backend.cube_pos_w[:, 2]).item()) > cfg.table_height
