# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native three-cube Franka stack slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFrankaStackRgbEnv,
    MacFrankaStackRgbEnvCfg,
    MacFrankaStackRgbTrainCfg,
    play_franka_stack_rgb_policy,
    replay_actions,
    rollout_env,
    train_franka_stack_rgb_policy,
)


def test_mac_franka_stack_rgb_reset_and_step_shapes():
    """The three-cube Franka stack env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaStackRgbEnvCfg(num_envs=8, seed=101, episode_length_s=0.5)
    env = MacFrankaStackRgbEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 42)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 42)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    assert env.sim_backend.state_dict()["task"] == "franka-stack-rgb"


def test_mac_franka_stack_rgb_can_complete_top_stack_release():
    """Opening the gripper over the final stack target should terminate successfully."""

    cfg = MacFrankaStackRgbEnvCfg(num_envs=2, seed=103, episode_length_s=0.5)
    env = MacFrankaStackRgbEnv(cfg)
    stack_offset = mx.array([0.0, 0.0, cfg.stack_offset_z], dtype=mx.float32)
    grasp_offset = mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)

    env.sim_backend.middle_stacked[:] = True
    env.sim_backend.top_stacked[:] = False
    env.sim_backend.middle_grasped[:] = False
    env.sim_backend.top_grasped[:] = True
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + 2.0 * cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.middle_cube_pos_w[:, :] = env.sim_backend.support_cube_pos_w + stack_offset
    env.sim_backend.top_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + grasp_offset
    env.sim_backend.state.joint_pos[:, 7] = 0.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert extras["reset_env_ids"] == [0, 1]
    assert extras["terminated_env_ids"] == [0, 1]
    assert bool(mx.all(extras["final_policy_observations"][:, 39] == 1.0).item())
    assert bool(mx.all(extras["final_policy_observations"][:, 40] == 1.0).item())
    assert bool(mx.all(extras["final_policy_observations"][:, 41] == 1.0).item())


def test_mac_franka_stack_rgb_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve three-cube Franka stack trajectories for fixed actions."""

    cfg = MacFrankaStackRgbEnvCfg(num_envs=4, seed=107, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFrankaStackRgbEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFrankaStackRgbEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_franka_stack_rgb_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable three-cube Franka stack checkpoint."""

    checkpoint_path = tmp_path / "franka_stack_rgb_policy.npz"
    train_cfg = MacFrankaStackRgbTrainCfg(
        env=MacFrankaStackRgbEnvCfg(num_envs=8, seed=109, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_franka_stack_rgb_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space

    episode_returns = play_franka_stack_rgb_policy(
        str(checkpoint_path),
        env_cfg=MacFrankaStackRgbEnvCfg(num_envs=1, seed=109, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_franka_stack_rgb_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    """Resuming three-cube Franka stack should preserve the checkpoint architecture."""

    first_checkpoint = tmp_path / "franka_stack_rgb_policy_initial.npz"
    resumed_checkpoint = tmp_path / "franka_stack_rgb_policy_resumed.npz"

    initial_result = train_franka_stack_rgb_policy(
        MacFrankaStackRgbTrainCfg(
            env=MacFrankaStackRgbEnvCfg(num_envs=8, seed=113, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_franka_stack_rgb_policy(
        MacFrankaStackRgbTrainCfg(
            env=MacFrankaStackRgbEnvCfg(num_envs=8, seed=113, episode_length_s=0.5),
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
