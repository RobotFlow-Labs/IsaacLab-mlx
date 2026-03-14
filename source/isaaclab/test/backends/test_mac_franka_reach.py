# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native Franka reach slice."""

from __future__ import annotations

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import MacFrankaReachEnv, MacFrankaReachEnvCfg, replay_actions, rollout_env  # noqa: E402


def test_mac_franka_reach_reset_and_step_shapes():
    """The Franka reach env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaReachEnvCfg(num_envs=8, seed=23, episode_length_s=0.5)
    env = MacFrankaReachEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 23)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 23)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    assert env.sim_backend.state_dict()["task"] == "franka-reach"


def test_mac_franka_reach_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve Franka reach trajectories for fixed actions."""

    cfg = MacFrankaReachEnvCfg(num_envs=4, seed=29, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFrankaReachEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFrankaReachEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()
