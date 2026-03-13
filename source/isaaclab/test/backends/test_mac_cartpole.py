# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native MLX cartpole reference path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacCartpoleEnv,
    MacCartpoleEnvCfg,
    MacCartpoleTrainCfg,
    play_cartpole_policy,
    train_cartpole_policy,
)


def test_mac_cartpole_env_step_shapes():
    """The vectorized mac cartpole env should expose the expected IsaacLab-style tensors."""
    env = MacCartpoleEnv(MacCartpoleEnvCfg(num_envs=8, seed=7, episode_length_s=0.3))

    obs, extras = env.reset()

    assert obs["policy"].shape == (8, 4)
    assert extras == {}
    assert env.runtime.compute_backend == "mlx"
    assert env.runtime.sim_backend == "mac-sim"

    next_obs, reward, terminated, truncated, step_extras = env.step([[1.0]] * env.num_envs)

    assert next_obs["policy"].shape == (8, 4)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)


def test_train_and_play_cartpole_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable checkpoint and sidecar metadata."""
    checkpoint_path = tmp_path / "cartpole_policy.npz"
    train_cfg = MacCartpoleTrainCfg(
        env=MacCartpoleEnvCfg(num_envs=8, seed=13, episode_length_s=0.3),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
    )

    result = train_cartpole_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["hidden_dim"] == 32

    episode_returns = play_cartpole_policy(
        str(checkpoint_path),
        env_cfg=MacCartpoleEnvCfg(num_envs=1, seed=13, episode_length_s=0.3),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)
