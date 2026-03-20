# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the reduced mac-native OpenArm manipulation slices."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacOpenArmBiReachEnv,
    MacOpenArmBiReachEnvCfg,
    MacOpenArmBiReachTrainCfg,
    MacOpenArmLiftEnv,
    MacOpenArmLiftEnvCfg,
    MacOpenArmLiftTrainCfg,
    MacOpenArmOpenDrawerEnv,
    MacOpenArmOpenDrawerEnvCfg,
    MacOpenArmOpenDrawerTrainCfg,
    MacOpenArmReachEnv,
    MacOpenArmReachEnvCfg,
    MacOpenArmReachTrainCfg,
    play_openarm_bi_reach_policy,
    play_openarm_lift_policy,
    play_openarm_open_drawer_policy,
    play_openarm_reach_policy,
    replay_actions,
    rollout_env,
    train_openarm_bi_reach_policy,
    train_openarm_lift_policy,
    train_openarm_open_drawer_policy,
    train_openarm_reach_policy,
)


def test_mac_openarm_reach_reset_and_step_shapes():
    cfg = MacOpenArmReachEnvCfg(num_envs=8, seed=23, episode_length_s=0.5)
    env = MacOpenArmReachEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 23)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 23)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "openarm-reach"
    assert state["semantic_contract"] == "reduced-openarm-surrogate"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_openarm_bi_reach_reset_and_step_shapes():
    cfg = MacOpenArmBiReachEnvCfg(num_envs=8, seed=29, episode_length_s=0.5)
    env = MacOpenArmBiReachEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 46)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 46)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "openarm-bi-reach"
    assert state["semantic_contract"] == "reduced-openarm-bimanual-surrogate"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_openarm_lift_reset_and_step_shapes():
    cfg = MacOpenArmLiftEnvCfg(num_envs=8, seed=31, episode_length_s=0.5)
    env = MacOpenArmLiftEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 27)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 27)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "openarm-lift"
    assert state["semantic_contract"] == "reduced-openarm-surrogate"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_openarm_open_drawer_reset_and_step_shapes():
    cfg = MacOpenArmOpenDrawerEnvCfg(num_envs=8, seed=37, episode_length_s=0.5)
    env = MacOpenArmOpenDrawerEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 28)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 28)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == "openarm-open-drawer"
    assert state["semantic_contract"] == "reduced-openarm-surrogate"
    assert state["upstream_alias_semantics_preserved"] is False


def test_mac_openarm_reach_rollout_replay_is_deterministic():
    cfg = MacOpenArmReachEnvCfg(num_envs=4, seed=41, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacOpenArmReachEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacOpenArmReachEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_mac_openarm_bi_reach_rollout_replay_is_deterministic():
    cfg = MacOpenArmBiReachEnvCfg(num_envs=4, seed=43, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacOpenArmBiReachEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacOpenArmBiReachEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_openarm_reach_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "openarm_reach_policy.npz"
    train_cfg = MacOpenArmReachTrainCfg(
        env=MacOpenArmReachEnvCfg(num_envs=8, seed=47, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_openarm_reach_policy(train_cfg)

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Reach-OpenArm-v0"
    assert metadata["hidden_dim"] == 32
    returns = play_openarm_reach_policy(
        str(checkpoint_path),
        env_cfg=MacOpenArmReachEnvCfg(num_envs=1, seed=47, episode_length_s=0.5),
        episodes=1,
    )
    assert len(returns) == 1
    assert isinstance(returns[0], float)


def test_train_and_play_openarm_bi_reach_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "openarm_bi_reach_policy.npz"
    train_cfg = MacOpenArmBiReachTrainCfg(
        env=MacOpenArmBiReachEnvCfg(num_envs=8, seed=53, episode_length_s=0.5),
        hidden_dim=48,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_openarm_bi_reach_policy(train_cfg)

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Reach-OpenArm-Bi-v0"
    assert metadata["hidden_dim"] == 48
    returns = play_openarm_bi_reach_policy(
        str(checkpoint_path),
        env_cfg=MacOpenArmBiReachEnvCfg(num_envs=1, seed=53, episode_length_s=0.5),
        episodes=1,
    )
    assert len(returns) == 1
    assert isinstance(returns[0], float)


def test_train_and_play_openarm_lift_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "openarm_lift_policy.npz"
    train_cfg = MacOpenArmLiftTrainCfg(
        env=MacOpenArmLiftEnvCfg(num_envs=8, seed=59, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_openarm_lift_policy(train_cfg)

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Lift-Cube-OpenArm-v0"
    returns = play_openarm_lift_policy(
        str(checkpoint_path),
        env_cfg=MacOpenArmLiftEnvCfg(num_envs=1, seed=59, episode_length_s=0.5),
        episodes=1,
    )
    assert len(returns) == 1
    assert isinstance(returns[0], float)


def test_train_and_play_openarm_open_drawer_smoke(tmp_path: Path):
    checkpoint_path = tmp_path / "openarm_open_drawer_policy.npz"
    train_cfg = MacOpenArmOpenDrawerTrainCfg(
        env=MacOpenArmOpenDrawerEnvCfg(num_envs=8, seed=61, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_openarm_open_drawer_policy(train_cfg)

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["task_id"] == "Isaac-Open-Drawer-OpenArm-v0"
    returns = play_openarm_open_drawer_policy(
        str(checkpoint_path),
        env_cfg=MacOpenArmOpenDrawerEnvCfg(num_envs=1, seed=61, episode_length_s=0.5),
        episodes=1,
    )
    assert len(returns) == 1
    assert isinstance(returns[0], float)
