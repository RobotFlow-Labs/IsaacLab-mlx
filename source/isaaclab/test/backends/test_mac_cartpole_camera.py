# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native synthetic cartpole camera slices."""

from __future__ import annotations

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacCartpoleCameraEnv,
    MacCartpoleDepthCameraEnvCfg,
    MacCartpoleRGBCameraEnvCfg,
    mac_env_diagnostics,
)


def test_mac_cartpole_rgb_camera_reset_and_step_shapes():
    """The RGB cartpole camera env should expose deterministic image observations."""

    cfg = MacCartpoleRGBCameraEnvCfg(num_envs=8, seed=17, episode_length_s=0.5)
    env = MacCartpoleCameraEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 100, 100, 3)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 100, 100, 3)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    diagnostics = mac_env_diagnostics(env)
    assert diagnostics["sensor"]["implementation"] == "analytic-cartpole-camera"
    assert diagnostics["sensor"]["camera_mode"] == "rgb"


def test_mac_cartpole_depth_camera_reset_and_step_shapes():
    """The depth cartpole camera env should expose deterministic single-channel image observations."""

    cfg = MacCartpoleDepthCameraEnvCfg(num_envs=4, seed=19, episode_length_s=0.5)
    env = MacCartpoleCameraEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (4, 100, 100, 1)
    assert float(mx.max(obs["policy"]).item()) <= cfg.camera_max_depth_m


def test_mac_cartpole_camera_is_deterministic_for_matching_seed():
    """The synthetic camera renderer should match exactly for identical seeds and actions."""

    cfg = MacCartpoleRGBCameraEnvCfg(num_envs=2, seed=23, episode_length_s=0.5)
    env_a = MacCartpoleCameraEnv(cfg)
    env_b = MacCartpoleCameraEnv(cfg)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_a, _, _, _, _ = env_a.step(actions)
    next_b, _, _, _, _ = env_b.step(actions)

    assert mx.array_equal(obs_a["policy"], obs_b["policy"]).item() is True
    assert mx.array_equal(next_a["policy"], next_b["policy"]).item() is True
