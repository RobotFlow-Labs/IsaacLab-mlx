# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native Franka teddy-bear lift slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFrankaTeddyBearLiftEnv,
    MacFrankaTeddyBearLiftEnvCfg,
    MacFrankaTeddyBearLiftTrainCfg,
    play_franka_teddy_bear_lift_policy,
    replay_actions,
    rollout_env,
    train_franka_teddy_bear_lift_policy,
)


def test_mac_franka_teddy_bear_lift_reset_and_step_shapes():
    """The Franka teddy-bear lift env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaTeddyBearLiftEnvCfg(num_envs=8, seed=61, episode_length_s=0.5)
    env = MacFrankaTeddyBearLiftEnv(cfg)

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
    assert env.sim_backend.state_dict()["task"] == "franka-teddy-bear-lift"


def test_mac_franka_teddy_bear_lift_can_grasp_and_raise_object():
    """Closing the gripper near the teddy-bear proxy should engage grasp logic and move it upward."""

    cfg = MacFrankaTeddyBearLiftEnvCfg(num_envs=2, seed=67, episode_length_s=0.5, lift_success_height=1.0)
    env = MacFrankaTeddyBearLiftEnv(cfg)
    env.sim_backend.cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array([0.0, 0.0, -0.03], dtype=mx.float32)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = -1.0
    _, _, _, _, _ = env.step(actions)

    assert bool(mx.any(env.sim_backend.grasped).item())
    assert float(mx.max(env.sim_backend.cube_pos_w[:, 2]).item()) > cfg.table_height


def test_mac_franka_teddy_bear_lift_success_snapshot_uses_pre_reset_observations():
    """Successful teddy-bear lift steps should expose pre-reset terminal observations for PPO bootstrapping."""

    cfg = MacFrankaTeddyBearLiftEnvCfg(num_envs=2, seed=69, episode_length_s=0.5)
    env = MacFrankaTeddyBearLiftEnv(cfg)
    env.sim_backend.grasped[:] = True
    env.sim_backend.cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    env.sim_backend.cube_pos_w[:, 2] = cfg.lift_success_height + 0.01

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert extras["reset_env_ids"] == [0, 1]
    assert extras["terminated_env_ids"] == [0, 1]
    final_obs = mx.array(extras["final_policy_observations"])
    assert final_obs.shape == (2, cfg.observation_space)
    assert float(mx.min(final_obs[:, 21]).item()) >= cfg.lift_success_height
    assert float(mx.max(final_obs[:, 25]).item()) <= 0.0


def test_mac_franka_teddy_bear_lift_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve teddy-bear lift trajectories for fixed actions."""

    cfg = MacFrankaTeddyBearLiftEnvCfg(num_envs=4, seed=71, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFrankaTeddyBearLiftEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFrankaTeddyBearLiftEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_franka_teddy_bear_lift_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable teddy-bear lift checkpoint."""

    checkpoint_path = tmp_path / "franka_teddy_bear_lift_policy.npz"
    train_cfg = MacFrankaTeddyBearLiftTrainCfg(
        env=MacFrankaTeddyBearLiftEnvCfg(num_envs=8, seed=73, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_franka_teddy_bear_lift_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space

    episode_returns = play_franka_teddy_bear_lift_policy(
        str(checkpoint_path),
        env_cfg=MacFrankaTeddyBearLiftEnvCfg(num_envs=1, seed=73, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_franka_teddy_bear_lift_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    """Resuming teddy-bear lift should preserve the checkpoint architecture."""

    first_checkpoint = tmp_path / "franka_teddy_bear_lift_policy_initial.npz"
    resumed_checkpoint = tmp_path / "franka_teddy_bear_lift_policy_resumed.npz"

    initial_result = train_franka_teddy_bear_lift_policy(
        MacFrankaTeddyBearLiftTrainCfg(
            env=MacFrankaTeddyBearLiftEnvCfg(num_envs=8, seed=79, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_franka_teddy_bear_lift_policy(
        MacFrankaTeddyBearLiftTrainCfg(
            env=MacFrankaTeddyBearLiftEnvCfg(num_envs=8, seed=79, episode_length_s=0.5),
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
