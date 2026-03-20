# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for reduced mac-native Agibot place slices."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacAgibotPlaceToy2BoxEnv,
    MacAgibotPlaceToy2BoxEnvCfg,
    MacAgibotPlaceToy2BoxTrainCfg,
    MacAgibotPlaceUprightMugEnv,
    MacAgibotPlaceUprightMugEnvCfg,
    MacAgibotPlaceUprightMugTrainCfg,
    play_agibot_place_toy2box_policy,
    play_agibot_place_upright_mug_policy,
    replay_actions,
    rollout_env,
    train_agibot_place_toy2box_policy,
    train_agibot_place_upright_mug_policy,
)


def test_mac_agibot_place_reset_and_step_shapes():
    cfg = MacAgibotPlaceToy2BoxEnvCfg(num_envs=8, seed=131, episode_length_s=0.5)
    env = MacAgibotPlaceToy2BoxEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 34)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 34)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "agibot-place-toy2box"
    assert state["semantic_contract"] == "reduced-agibot-place-surrogate"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_agibot_upright_mug_reset_and_step_shapes():
    cfg = MacAgibotPlaceUprightMugEnvCfg(num_envs=8, seed=137, episode_length_s=0.5)
    env = MacAgibotPlaceUprightMugEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 34)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 34)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "agibot-place-upright-mug"
    assert state["semantic_contract"] == "reduced-agibot-place-surrogate"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_agibot_place_rollout_replay_is_deterministic():
    cfg = MacAgibotPlaceToy2BoxEnvCfg(num_envs=4, seed=139, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacAgibotPlaceToy2BoxEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacAgibotPlaceToy2BoxEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_agibot_place_toy2box_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "agibot_place_toy2box_policy.npz"
    train_cfg = MacAgibotPlaceToy2BoxTrainCfg(
        env=MacAgibotPlaceToy2BoxEnvCfg(num_envs=8, seed=149, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_agibot_place_toy2box_policy(train_cfg)

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
    returns = play_agibot_place_toy2box_policy(
        str(checkpoint_path),
        env_cfg=MacAgibotPlaceToy2BoxEnvCfg(num_envs=1, seed=149, episode_length_s=0.5),
        episodes=1,
    )
    assert len(returns) == 1
    assert isinstance(returns[0], float)


def test_train_and_play_agibot_place_upright_mug_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "agibot_place_upright_mug_policy.npz"
    train_cfg = MacAgibotPlaceUprightMugTrainCfg(
        env=MacAgibotPlaceUprightMugEnvCfg(num_envs=8, seed=151, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_agibot_place_upright_mug_policy(train_cfg)

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0"
    returns = play_agibot_place_upright_mug_policy(
        str(checkpoint_path),
        env_cfg=MacAgibotPlaceUprightMugEnvCfg(num_envs=1, seed=151, episode_length_s=0.5),
        episodes=1,
    )
    assert len(returns) == 1
    assert isinstance(returns[0], float)
