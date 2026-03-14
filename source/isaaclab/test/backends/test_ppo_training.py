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
    play_categorical_policy_checkpoint,
    play_gaussian_policy_checkpoint,
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


class _DummyPolicyCfg:
    observation_space = 3
    action_space = 2


class _DummyPolicyEnv:
    def __init__(self, cfg: _DummyPolicyCfg):
        self.cfg = cfg
        self.max_episode_length = 4
        self._steps = 0

    def reset(self):
        self._steps = 0
        return {"policy": mx.zeros((1, self.cfg.observation_space), dtype=mx.float32)}, {}

    def step(self, actions):
        del actions
        self._steps += 1
        return (
            {"policy": mx.full((1, self.cfg.observation_space), float(self._steps), dtype=mx.float32)},
            mx.zeros((1,), dtype=mx.float32),
            mx.zeros((1,), dtype=mx.bool_),
            mx.zeros((1,), dtype=mx.bool_),
            {"completed_returns": [float(self._steps)]},
        )


class _DummyGaussianModel:
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.loaded_path: str | None = None

    def load_weights(self, path: str) -> None:
        self.loaded_path = path

    def __call__(self, obs):
        batch = obs.shape[0]
        return mx.ones((batch, self.action_dim), dtype=mx.float32), mx.zeros((batch,), dtype=mx.float32)


class _DummyCategoricalModel:
    def __init__(self, obs_dim: int, hidden_dim: int):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.loaded_path: str | None = None

    def load_weights(self, path: str) -> None:
        self.loaded_path = path

    def __call__(self, obs):
        batch = obs.shape[0]
        logits = mx.array([[0.0, 1.0]] * batch, dtype=mx.float32)
        return logits, mx.zeros((batch,), dtype=mx.float32)


def test_play_gaussian_policy_checkpoint_collects_episode_returns(tmp_path: Path):
    checkpoint = tmp_path / "gaussian_policy.npz"
    write_checkpoint_metadata(
        checkpoint,
        build_checkpoint_metadata(
            hidden_dim=64,
            observation_space=3,
            action_space=2,
            task_id="dummy-gaussian",
            policy_distribution="gaussian",
            train_cfg={"updates": 1},
        ),
    )

    episode_returns = play_gaussian_policy_checkpoint(
        checkpoint,
        env_factory=_DummyPolicyEnv,
        env_cfg=_DummyPolicyCfg(),
        model_factory=_DummyGaussianModel,
        default_hidden_dim=32,
        episodes=2,
    )

    assert episode_returns == [1.0, 2.0]


def test_play_categorical_policy_checkpoint_collects_episode_returns(tmp_path: Path):
    checkpoint = tmp_path / "categorical_policy.npz"
    write_checkpoint_metadata(
        checkpoint,
        build_checkpoint_metadata(
            hidden_dim=48,
            observation_space=3,
            action_space=1,
            task_id="dummy-categorical",
            policy_distribution="categorical",
            policy_action_space=2,
            train_cfg={"updates": 1},
        ),
    )

    episode_returns = play_categorical_policy_checkpoint(
        checkpoint,
        env_factory=_DummyPolicyEnv,
        env_cfg=_DummyPolicyCfg(),
        model_factory=_DummyCategoricalModel,
        action_transform=lambda actions: actions.reshape((-1, 1)),
        default_hidden_dim=16,
        episodes=3,
    )

    assert episode_returns == [1.0, 2.0, 3.0]
