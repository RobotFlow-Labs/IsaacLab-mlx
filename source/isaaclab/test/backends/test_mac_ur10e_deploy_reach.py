# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the reduced mac-native UR10e deploy-reach slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacUR10eDeployReachEnv,
    MacUR10eDeployReachEnvCfg,
    MacUR10eDeployReachTrainCfg,
    play_ur10e_deploy_reach_policy,
    replay_actions,
    rollout_env,
    train_ur10e_deploy_reach_policy,
)


def test_mac_ur10e_deploy_reach_reset_and_step_shapes():
    """The reduced UR10e deploy-reach env should expose deterministic IsaacLab-style tensors."""

    cfg = MacUR10eDeployReachEnvCfg(num_envs=8, seed=23, episode_length_s=0.5)
    env = MacUR10eDeployReachEnv(cfg)

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
    assert state["task"] == "ur10e-deploy-reach"
    assert state["semantic_contract"] == "reduced-analytic-pose"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_ur10e_deploy_reach_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve UR10e deploy-reach traces for fixed actions."""

    cfg = MacUR10eDeployReachEnvCfg(num_envs=4, seed=29, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacUR10eDeployReachEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacUR10eDeployReachEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_ur10e_deploy_reach_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable UR10e deploy-reach checkpoint."""

    checkpoint_path = tmp_path / "ur10e_deploy_reach_policy.npz"
    train_cfg = MacUR10eDeployReachTrainCfg(
        env=MacUR10eDeployReachEnvCfg(num_envs=8, seed=23, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_ur10e_deploy_reach_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Deploy-Reach-UR10e-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space
    assert metadata["observation_space"] == train_cfg.env.observation_space

    episode_returns = play_ur10e_deploy_reach_policy(
        str(checkpoint_path),
        env_cfg=MacUR10eDeployReachEnvCfg(num_envs=1, seed=23, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_ur10e_deploy_reach_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    """Resuming UR10e deploy-reach should preserve the checkpoint architecture."""

    first_checkpoint = tmp_path / "ur10e_deploy_reach_policy_initial.npz"
    resumed_checkpoint = tmp_path / "ur10e_deploy_reach_policy_resumed.npz"

    initial_result = train_ur10e_deploy_reach_policy(
        MacUR10eDeployReachTrainCfg(
            env=MacUR10eDeployReachEnvCfg(num_envs=8, seed=29, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_ur10e_deploy_reach_policy(
        MacUR10eDeployReachTrainCfg(
            env=MacUR10eDeployReachEnvCfg(num_envs=8, seed=29, episode_length_s=0.5),
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
