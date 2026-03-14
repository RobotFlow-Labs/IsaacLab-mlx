# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native ANYmal-C flat locomotion slice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacAnymalCFlatEnv,
    MacAnymalCFlatEnvCfg,
    MacAnymalCTrainCfg,
    play_anymal_c_policy,
    replay_actions,
    rollout_env,
    train_anymal_c_policy,
)


def test_mac_anymal_c_reset_and_step_shapes():
    """The flat ANYmal-C env should expose the expected IsaacLab-style tensors."""
    cfg = MacAnymalCFlatEnvCfg(num_envs=8, seed=17, episode_length_s=0.5)
    env = MacAnymalCFlatEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 48)
    assert env.runtime.compute_backend == "mlx"
    assert env.runtime.sim_backend == "mac-sim"
    assert env.sim_backend.terrain.state_dict()["type"] == "plane"
    assert env.sim_backend.contact_model.state_dict()["tracked_bodies"][0] == "base"

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 48)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)


def test_mac_anymal_c_base_contact_termination():
    """Dropping the base onto the terrain should terminate all environments."""
    cfg = MacAnymalCFlatEnvCfg(num_envs=4, seed=5, episode_length_s=0.5)
    env = MacAnymalCFlatEnv(cfg)

    env.sim_backend.root_pos_w[:, 2] = 0.0
    env.sim_backend._last_body_pos_w = env.sim_backend._body_positions()
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert len(extras["completed_lengths"]) == cfg.num_envs
    assert len(extras["completed_returns"]) == cfg.num_envs


def test_anymal_c_rollout_replay_is_deterministic():
    """Shared rollout/replay helpers should preserve the ANYmal-C trajectory for fixed actions."""
    cfg = MacAnymalCFlatEnvCfg(num_envs=4, seed=11, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacAnymalCFlatEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacAnymalCFlatEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_anymal_c_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable ANYmal-C checkpoint."""
    checkpoint_path = tmp_path / "anymal_c_flat_policy.npz"
    train_cfg = MacAnymalCTrainCfg(
        env=MacAnymalCFlatEnvCfg(num_envs=8, seed=23, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_anymal_c_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["hidden_dim"] == 32

    episode_returns = play_anymal_c_policy(
        str(checkpoint_path),
        env_cfg=MacAnymalCFlatEnvCfg(num_envs=1, seed=23, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)
