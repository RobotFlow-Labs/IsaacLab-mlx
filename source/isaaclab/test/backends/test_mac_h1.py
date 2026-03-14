# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native H1 flat locomotion slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacH1FlatEnv,
    MacH1FlatEnvCfg,
    MacH1RoughEnv,
    MacH1RoughEnvCfg,
    MacH1TrainCfg,
    play_h1_policy,
    replay_actions,
    rollout_env,
    train_h1_policy,
)


def test_mac_h1_reset_and_step_shapes():
    """The flat H1 env should expose the expected IsaacLab-style tensors."""
    cfg = MacH1FlatEnvCfg(num_envs=8, seed=17, episode_length_s=0.5)
    env = MacH1FlatEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 69)
    assert env.runtime.compute_backend == "mlx"
    assert env.runtime.sim_backend == "mac-sim"
    assert env.sim_backend.terrain.state_dict()["type"] == "plane"
    assert env.sim_backend.contact_model.state_dict()["tracked_bodies"][0] == "torso_link"

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 69)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)


def test_mac_h1_height_scan_expands_observation_space():
    """Enabling the height scan should append deterministic sensor channels to the policy observation."""
    cfg = MacH1FlatEnvCfg(
        num_envs=4,
        seed=21,
        episode_length_s=0.5,
        height_scan_enabled=True,
        height_scan_offsets=((-0.25, 0.0), (0.0, 0.0), (0.25, 0.0)),
    )
    env = MacH1FlatEnv(cfg)

    obs, _ = env.reset()

    assert env.height_scan_sensor is not None
    assert env.height_scan_dim == 3
    assert obs["policy"].shape == (4, 72)


def test_mac_h1_rough_reset_and_step_shapes():
    """The rough H1 env should expose wave terrain and height-scan channels deterministically."""
    cfg = MacH1RoughEnvCfg(num_envs=6, seed=19, episode_length_s=0.5)
    env = MacH1RoughEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert env.sim_backend.terrain.state_dict()["type"] == "wave"
    assert env.height_scan_sensor is not None
    assert env.height_scan_dim == 9
    assert obs["policy"].shape == (6, 78)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (6, 78)
    assert reward.shape == (6,)
    assert terminated.shape == (6,)
    assert truncated.shape == (6,)
    assert isinstance(step_extras, dict)


def test_mac_h1_rough_rollout_replay_is_deterministic():
    """Shared rollout/replay helpers should preserve the rough H1 trajectory for fixed actions."""
    cfg = MacH1RoughEnvCfg(num_envs=4, seed=31, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacH1RoughEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacH1RoughEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert bool(mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item())


def test_mac_h1_base_contact_termination():
    """Dropping the torso onto the terrain should terminate all environments."""
    cfg = MacH1FlatEnvCfg(num_envs=4, seed=5, episode_length_s=0.5, min_root_height=0.0)
    env = MacH1FlatEnv(cfg)

    env.sim_backend.root_pos_w[:, 2] = 0.01
    env.sim_backend._last_body_pos_w = env.sim_backend._body_positions()
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    _, _, terminated, truncated, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert not bool(mx.any(truncated).item())
    assert len(extras["completed_lengths"]) == cfg.num_envs
    assert len(extras["completed_returns"]) == cfg.num_envs


def test_h1_rollout_replay_is_deterministic():
    """Shared rollout/replay helpers should preserve the H1 trajectory for fixed actions."""
    cfg = MacH1FlatEnvCfg(num_envs=4, seed=11, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacH1FlatEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacH1FlatEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert bool(mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item())


def test_train_and_play_h1_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable H1 checkpoint."""
    checkpoint_path = tmp_path / "h1_flat_policy.npz"
    train_cfg = MacH1TrainCfg(
        env=MacH1FlatEnvCfg(num_envs=8, seed=23, episode_length_s=0.5),
        hidden_dim=64,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_h1_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Velocity-Flat-H1-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 64
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space

    episode_returns = play_h1_policy(
        str(checkpoint_path),
        env_cfg=MacH1FlatEnvCfg(num_envs=1, seed=23, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_h1_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    """Resuming H1 should preserve the checkpoint architecture instead of the new cfg hidden size."""
    first_checkpoint = tmp_path / "h1_flat_policy_initial.npz"
    resumed_checkpoint = tmp_path / "h1_flat_policy_resumed.npz"

    initial_result = train_h1_policy(
        MacH1TrainCfg(
            env=MacH1FlatEnvCfg(num_envs=8, seed=29, episode_length_s=0.5),
            hidden_dim=64,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_h1_policy(
        MacH1TrainCfg(
            env=MacH1FlatEnvCfg(num_envs=8, seed=29, episode_length_s=0.5),
            hidden_dim=192,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(resumed_checkpoint),
            resume_from=initial_result["checkpoint_path"],
            eval_interval=1,
        )
    )

    metadata = json.loads(Path(resumed_result["metadata_path"]).read_text(encoding="utf-8"))
    assert resumed_result["resumed_from"] == initial_result["checkpoint_path"]
    assert metadata["metadata_version"] == 2
    assert metadata["hidden_dim"] == 64


def test_train_h1_accepts_rough_env_cfg(tmp_path: Path):
    """The shared H1 trainer should size itself from the rough runtime observation width."""
    checkpoint_path = tmp_path / "h1_rough_policy.npz"
    train_cfg = MacH1TrainCfg(
        env=MacH1RoughEnvCfg(num_envs=8, seed=41, episode_length_s=0.5),
        hidden_dim=64,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_h1_policy(train_cfg)
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))

    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Velocity-Rough-H1-v0"
    assert metadata["observation_space"] == 78


def test_play_h1_infers_rough_env_cfg_from_checkpoint_metadata(tmp_path: Path):
    """H1 replay without an explicit env cfg should recover the rough task shape from checkpoint metadata."""

    checkpoint_path = tmp_path / "h1_rough_policy.npz"
    train_h1_policy(
        MacH1TrainCfg(
            env=MacH1RoughEnvCfg(num_envs=8, seed=43, episode_length_s=0.5),
            hidden_dim=64,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(checkpoint_path),
            eval_interval=1,
        )
    )

    episode_returns = play_h1_policy(str(checkpoint_path), episodes=1)

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_h1_rough_uses_rough_default_checkpoint_path(tmp_path: Path, monkeypatch):
    """Rough H1 training should not clobber the flat default checkpoint path when no override is provided."""

    monkeypatch.chdir(tmp_path)
    result = train_h1_policy(
        MacH1TrainCfg(
            env=MacH1RoughEnvCfg(num_envs=8, seed=47, episode_length_s=0.5),
            hidden_dim=64,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            eval_interval=1,
        )
    )

    assert result["checkpoint_path"].endswith("logs/mlx/h1_rough_policy.npz")
    assert Path(result["checkpoint_path"]).exists()
