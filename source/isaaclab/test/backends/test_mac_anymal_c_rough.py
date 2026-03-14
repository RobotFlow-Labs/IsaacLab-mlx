# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native rough ANYmal-C slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacAnymalCRoughEnv,
    MacAnymalCRoughEnvCfg,
    MacAnymalCTrainCfg,
    play_anymal_c_policy,
    replay_actions,
    rollout_env,
    train_anymal_c_policy,
)


def test_mac_anymal_c_rough_reset_and_height_scan_shape():
    """The rough ANYmal-C env should enable the raycast-backed height scan by default."""

    cfg = MacAnymalCRoughEnvCfg(num_envs=6, seed=17, episode_length_s=0.5)
    env = MacAnymalCRoughEnv(cfg)

    obs, extras = env.reset()

    assert extras == {}
    assert env.height_scan_sensor is not None
    assert env.height_scan_dim == 9
    assert env.sim_backend.terrain.state_dict()["type"] == "wave"
    assert obs["policy"].shape == (6, 57)
    assert env.height_scan_sensor.state_dict()["terrain_type"] == "wave"


def test_mac_anymal_c_rough_rollout_replay_is_deterministic():
    """Rollout and replay helpers should preserve the rough ANYmal-C trajectory for fixed actions."""

    cfg = MacAnymalCRoughEnvCfg(num_envs=4, seed=19, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacAnymalCRoughEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacAnymalCRoughEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_anymal_c_accepts_rough_env_cfg(tmp_path: Path):
    """The shared ANYmal-C trainer should size itself from the rough runtime observation width."""

    checkpoint_path = tmp_path / "anymal_c_rough_policy.npz"
    train_cfg = MacAnymalCTrainCfg(
        env=MacAnymalCRoughEnvCfg(num_envs=8, seed=41, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_anymal_c_policy(train_cfg)
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))

    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Velocity-Rough-Anymal-C-Direct-v0"
    assert metadata["observation_space"] == 57


def test_play_anymal_c_infers_rough_env_cfg_from_checkpoint_metadata(tmp_path: Path):
    """ANYmal-C replay without an explicit env cfg should recover the rough task shape from checkpoint metadata."""

    checkpoint_path = tmp_path / "anymal_c_rough_policy.npz"
    train_anymal_c_policy(
        MacAnymalCTrainCfg(
            env=MacAnymalCRoughEnvCfg(num_envs=8, seed=43, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(checkpoint_path),
            eval_interval=1,
        )
    )

    episode_returns = play_anymal_c_policy(str(checkpoint_path), episodes=1)

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_anymal_c_rough_uses_rough_default_checkpoint_path(tmp_path: Path, monkeypatch):
    """Rough ANYmal-C training should not clobber the flat default checkpoint path when no override is provided."""

    monkeypatch.chdir(tmp_path)
    result = train_anymal_c_policy(
        MacAnymalCTrainCfg(
            env=MacAnymalCRoughEnvCfg(num_envs=8, seed=47, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            eval_interval=1,
        )
    )

    assert result["checkpoint_path"].endswith("logs/mlx/anymal_c_rough_policy.npz")
    assert Path(result["checkpoint_path"]).exists()
