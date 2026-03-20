# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the reduced mac-native Factory peg-insert slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFactoryPegInsertEnv,
    MacFactoryPegInsertEnvCfg,
    MacFactoryPegInsertTrainCfg,
    play_factory_peg_insert_policy,
    replay_actions,
    rollout_env,
    train_factory_peg_insert_policy,
)


def test_mac_factory_peg_insert_reset_and_step_shapes():
    """The Factory peg-insert env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFactoryPegInsertEnvCfg(num_envs=8, seed=173, episode_length_s=0.5)
    env = MacFactoryPegInsertEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, cfg.observation_space)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, cfg.observation_space)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)

    state = env.sim_backend.state_dict()
    assert state["task"] == "factory-peg-insert"
    assert state["upstream_task_id"] == "Isaac-Factory-PegInsert-Direct-v0"
    assert state["semantic_contract"] == "reduced-analytic-peg-insert"
    assert state["upstream_alias_semantics_preserved"] is False
    assert state["gripper_variant"] == "peg-insert"
    assert state["subsystems"]["peg_insert_logic"] is True
    assert state["subsystems"]["peg_variant_randomization"] is True


def test_mac_factory_peg_insert_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve Factory peg-insert trajectories for fixed actions."""

    cfg = MacFactoryPegInsertEnvCfg(num_envs=4, seed=179, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFactoryPegInsertEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFactoryPegInsertEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_factory_peg_insert_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable Factory peg-insert checkpoint."""

    checkpoint_path = tmp_path / "factory_peg_insert_policy.npz"
    train_cfg = MacFactoryPegInsertTrainCfg(
        env=MacFactoryPegInsertEnvCfg(num_envs=8, seed=181, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_factory_peg_insert_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Factory-PegInsert-Direct-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space
    assert metadata["semantic_contract"] == "reduced-analytic-peg-insert"
    assert metadata["upstream_alias_semantics_preserved"] is False

    episode_returns = play_factory_peg_insert_policy(
        str(checkpoint_path),
        env_cfg=MacFactoryPegInsertEnvCfg(num_envs=1, seed=181, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)
