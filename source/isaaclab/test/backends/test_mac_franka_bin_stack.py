# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native bin-anchored three-cube Franka stack slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFrankaBinStackEnv,
    MacFrankaBinStackEnvCfg,
    MacFrankaBinStackTrainCfg,
    play_franka_bin_stack_policy,
    replay_actions,
    rollout_env,
    train_franka_bin_stack_policy,
)


def test_mac_franka_bin_stack_reset_and_step_shapes():
    """The bin-anchored three-cube Franka stack env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaBinStackEnvCfg(num_envs=8, seed=131, episode_length_s=0.5)
    env = MacFrankaBinStackEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 45)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 45)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    assert env.sim_backend.state_dict()["task"] == "franka-bin-stack"
    assert env.sim_backend.state_dict()["semantic_contract"] == "reduced-no-mimic"
    assert env.sim_backend.state_dict()["upstream_alias_semantics_preserved"] is False
    assert env.sim_backend.state_dict()["subsystems"]["bin_anchor_observation_mode"] == "mirrored-support-anchor-tail"


def test_mac_franka_bin_stack_resets_support_cube_inside_bin_anchor():
    """The support cube should reset exactly on the bin anchor while the other cubes stay offset."""

    cfg = MacFrankaBinStackEnvCfg(num_envs=8, seed=137, episode_length_s=0.5)
    env = MacFrankaBinStackEnv(cfg)
    env.reset()

    assert bool(mx.allclose(env.sim_backend.support_cube_pos_w, env.sim_backend.bin_anchor_pos_w).item())
    middle_offset = mx.linalg.norm(env.sim_backend.middle_cube_pos_w - env.sim_backend.bin_anchor_pos_w, axis=1)
    top_offset = mx.linalg.norm(env.sim_backend.top_cube_pos_w - env.sim_backend.bin_anchor_pos_w, axis=1)
    assert bool(mx.all(middle_offset > 0.05).item())
    assert bool(mx.all(top_offset > 0.05).item())


def test_mac_franka_bin_stack_can_complete_top_stack_release():
    """Opening the gripper over the final top-stack target should terminate successfully."""

    cfg = MacFrankaBinStackEnvCfg(num_envs=2, seed=139, episode_length_s=0.5)
    env = MacFrankaBinStackEnv(cfg)
    stack_offset = mx.array([0.0, 0.0, cfg.stack_offset_z], dtype=mx.float32)
    grasp_offset = mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)

    env.sim_backend.middle_stacked[:] = True
    env.sim_backend.top_stacked[:] = False
    env.sim_backend.middle_grasped[:] = False
    env.sim_backend.top_grasped[:] = True
    env.sim_backend.bin_anchor_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + 2.0 * cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.bin_anchor_pos_w
    env.sim_backend.middle_cube_pos_w[:, :] = env.sim_backend.support_cube_pos_w + stack_offset
    env.sim_backend.top_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + grasp_offset
    env.sim_backend.state.joint_pos[:, 7] = 0.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert extras["reset_env_ids"] == [0, 1]
    assert extras["terminated_env_ids"] == [0, 1]
    assert extras["final_policy_observations"].shape == (2, 45)
    assert bool(mx.all(extras["final_policy_observations"][:, 40] == 1.0).item())
    assert bool(mx.all(extras["final_policy_observations"][:, 41] == 1.0).item())
    assert bool(mx.allclose(extras["final_policy_observations"][:, 25:28], extras["final_policy_observations"][:, 42:45]).item())


def test_mac_franka_bin_stack_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve bin-stack trajectories for fixed actions."""

    cfg = MacFrankaBinStackEnvCfg(num_envs=4, seed=149, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFrankaBinStackEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFrankaBinStackEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_franka_bin_stack_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable bin-stack Franka checkpoint."""

    checkpoint_path = tmp_path / "franka_bin_stack_policy.npz"
    train_cfg = MacFrankaBinStackTrainCfg(
        env=MacFrankaBinStackEnvCfg(num_envs=8, seed=151, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_franka_bin_stack_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space

    episode_returns = play_franka_bin_stack_policy(
        str(checkpoint_path),
        env_cfg=MacFrankaBinStackEnvCfg(num_envs=1, seed=151, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_franka_bin_stack_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    """Resuming the bin-stack task should preserve the checkpoint architecture."""

    first_checkpoint = tmp_path / "franka_bin_stack_initial.npz"
    resumed_checkpoint = tmp_path / "franka_bin_stack_resumed.npz"

    initial_result = train_franka_bin_stack_policy(
        MacFrankaBinStackTrainCfg(
            env=MacFrankaBinStackEnvCfg(num_envs=8, seed=157, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_franka_bin_stack_policy(
        MacFrankaBinStackTrainCfg(
            env=MacFrankaBinStackEnvCfg(num_envs=8, seed=157, episode_length_s=0.5),
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
