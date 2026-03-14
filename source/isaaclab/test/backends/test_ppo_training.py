# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared MLX PPO helper substrate."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from isaaclab.backends.mac_sim.ppo_training import (
    build_checkpoint_metadata,
    checkpoint_metadata_path,
    compute_gae,
    mean_recent_return,
    normalize_advantages,
    read_checkpoint_metadata,
    resolve_resume_hidden_dim,
    write_checkpoint_metadata,
)


def test_checkpoint_metadata_round_trip_and_resume_hidden_dim(tmp_path: Path):
    checkpoint = tmp_path / "policy.npz"
    metadata = build_checkpoint_metadata(
        hidden_dim=192,
        observation_space=69,
        action_space=19,
        task_id="Isaac-Velocity-Flat-H1-v0",
        policy_distribution="gaussian",
        action_std=0.28,
        train_cfg={"updates": 10},
    )

    metadata_path = write_checkpoint_metadata(checkpoint, metadata)

    assert metadata_path == checkpoint_metadata_path(checkpoint)
    round_trip = read_checkpoint_metadata(checkpoint)
    assert round_trip["metadata_version"] == 2
    assert round_trip["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert round_trip["task_id"] == "Isaac-Velocity-Flat-H1-v0"
    assert round_trip["policy_distribution"] == "gaussian"
    assert round_trip["hidden_dim"] == 192
    assert round_trip["policy_action_space"] == 19
    assert resolve_resume_hidden_dim(str(checkpoint), 128) == 192
    assert resolve_resume_hidden_dim(None, 128) == 128


def test_compute_gae_matches_manual_reference():
    rewards_t = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    dones_t = mx.array([[0.0], [0.0], [1.0]], dtype=mx.float32)
    values_t = mx.array([[0.2], [0.3], [0.4]], dtype=mx.float32)
    next_values_t = mx.array([[0.3], [0.4], [0.0]], dtype=mx.float32)

    advantages, returns = compute_gae(rewards_t, dones_t, values_t, next_values_t, gamma=0.99, gae_lambda=0.95)

    expected_advantages = mx.array([[5.36809], [4.5413], [2.6]], dtype=mx.float32)
    expected_returns = expected_advantages + values_t

    assert mx.allclose(advantages, expected_advantages, atol=1e-5)
    assert mx.allclose(returns, expected_returns, atol=1e-5)


def test_normalize_advantages_and_recent_return_helpers():
    normalized = normalize_advantages(mx.array([1.0, 2.0, 4.0], dtype=mx.float32))

    assert abs(float(mx.mean(normalized))) < 1e-6
    assert abs(float(mx.mean(mx.square(normalized))) - 1.0) < 1e-5
    assert mean_recent_return([1.0, 2.0, 3.0, 4.0], window=2) == 3.5
