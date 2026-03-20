# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the reduced mac-native UR10 reach slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacUR10ReachEnv,
    MacUR10ReachEnvCfg,
    MacUR10ReachTrainCfg,
    play_ur10_reach_policy,
    replay_actions,
    rollout_env,
    train_ur10_reach_policy,
)


def test_mac_ur10_reach_reset_and_step_shapes():
    cfg = MacUR10ReachEnvCfg(num_envs=8, seed=23, episode_length_s=0.5)
    env = MacUR10ReachEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 19)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 19)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "ur10-reach"
    assert state["semantic_contract"] == "reduced-analytic-pose"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_ur10_reach_rollout_replay_is_deterministic():
    cfg = MacUR10ReachEnvCfg(num_envs=4, seed=29, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacUR10ReachEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacUR10ReachEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_ur10_reach_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "ur10_reach_policy.npz"
    train_cfg = MacUR10ReachTrainCfg(
        env=MacUR10ReachEnvCfg(num_envs=8, seed=31, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_ur10_reach_policy(train_cfg)

    metadata_path = Path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["metadata_version"] == 2
    assert metadata["task_id"] == "Isaac-Reach-UR10-v0"
    assert metadata["hidden_dim"] == 32

    episode_returns = play_ur10_reach_policy(
        str(checkpoint_path),
        env_cfg=MacUR10ReachEnvCfg(num_envs=1, seed=31, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_ur10_reach_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    first_checkpoint = tmp_path / "ur10_reach_policy_initial.npz"
    resumed_checkpoint = tmp_path / "ur10_reach_policy_resumed.npz"

    initial_result = train_ur10_reach_policy(
        MacUR10ReachTrainCfg(
            env=MacUR10ReachEnvCfg(num_envs=8, seed=37, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_ur10_reach_policy(
        MacUR10ReachTrainCfg(
            env=MacUR10ReachEnvCfg(num_envs=8, seed=37, episode_length_s=0.5),
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
    assert metadata["hidden_dim"] == 32
