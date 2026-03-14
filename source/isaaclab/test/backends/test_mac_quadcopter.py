# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native quadcopter environment."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from isaaclab.backends.mac_sim import MacQuadcopterEnv, MacQuadcopterEnvCfg  # noqa: E402


def test_mac_quadcopter_reset_and_step_shapes():
    """The quadcopter env should expose IsaacLab-style vectorized tensors."""
    cfg = MacQuadcopterEnvCfg(num_envs=8, seed=17, episode_length_s=0.5)
    env = MacQuadcopterEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 12)
    assert env.runtime.compute_backend == "mlx"
    assert env.runtime.sim_backend == "mac-sim"

    actions = mx.zeros((cfg.num_envs, 4), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 12)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)


def test_mac_quadcopter_termination_on_height_violation():
    """A root-height violation should terminate and emit reset extras."""
    cfg = MacQuadcopterEnvCfg(num_envs=4, seed=9, episode_length_s=0.5)
    env = MacQuadcopterEnv(cfg)

    env.sim_backend.root_pos_w[:, 2] = cfg.min_height - 0.05
    actions = mx.zeros((cfg.num_envs, 4), dtype=mx.float32)
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert "completed_lengths" in extras
    assert "final_distance_to_goal" in extras
    assert len(extras["completed_lengths"]) == cfg.num_envs


def test_mac_quadcopter_root_state_io():
    """The backend root pose/velocity write APIs should update state buffers."""
    cfg = MacQuadcopterEnvCfg(num_envs=3, seed=3, episode_length_s=0.5)
    env = MacQuadcopterEnv(cfg)

    pose = mx.array(
        [
            [0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
            [0.3, -0.1, 1.1, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=mx.float32,
    )
    velocity = mx.array(
        [
            [0.4, 0.0, -0.2, 0.01, 0.02, 0.03],
            [0.0, -0.4, 0.1, -0.02, 0.01, 0.0],
        ],
        dtype=mx.float32,
    )

    env.sim_backend.write_root_pose(None, pose, env_ids=[0, 2])
    env.sim_backend.write_root_velocity(None, velocity, env_ids=[0, 2])

    assert mx.allclose(env.sim_backend.root_pos_w[[0, 2]], pose[:, :3]).item()
    assert mx.allclose(env.sim_backend.root_lin_vel_b[[0, 2]], velocity[:, :3]).item()
    assert mx.allclose(env.sim_backend.root_ang_vel_b[[0, 2]], velocity[:, 3:6]).item()
