# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native Franka instance-randomized stack slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFrankaStackInstanceRandomizeEnv,
    MacFrankaStackInstanceRandomizeEnvCfg,
    MacFrankaStackInstanceRandomizeTrainCfg,
    play_franka_stack_instance_randomize_policy,
    replay_actions,
    rollout_env,
    train_franka_stack_instance_randomize_policy,
)


def test_mac_franka_stack_instance_randomize_reset_and_step_shapes():
    """The instance-randomized Franka stack env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaStackInstanceRandomizeEnvCfg(num_envs=8, seed=73, episode_length_s=0.5)
    env = MacFrankaStackInstanceRandomizeEnv(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 35)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 35)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    assert env.sim_backend.state_dict()["task"] == "franka-stack-instance-randomize"


def test_mac_franka_stack_instance_randomize_tracks_variant_ids():
    """Variant observations should stay in-bounds and distinct across support/movable objects."""

    cfg = MacFrankaStackInstanceRandomizeEnvCfg(num_envs=8, seed=79, episode_length_s=0.5)
    env = MacFrankaStackInstanceRandomizeEnv(cfg)

    obs, _ = env.reset()
    variant_obs = obs["policy"][:, -2:]

    assert bool(mx.all(variant_obs >= -1.0).item())
    assert bool(mx.all(variant_obs <= 1.0).item())
    assert bool(mx.all(env.sim_backend.support_variant_id != env.sim_backend.movable_variant_id).item())


def test_mac_franka_stack_instance_randomize_can_release_on_support_cube():
    """Opening the gripper over the stack target should mark the instance-randomized task as stacked."""

    cfg = MacFrankaStackInstanceRandomizeEnvCfg(num_envs=2, seed=83, episode_length_s=0.5)
    env = MacFrankaStackInstanceRandomizeEnv(cfg)
    env.sim_backend.cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.grasped[:] = True
    env.sim_backend.stacked[:] = False
    env.sim_backend.state.joint_pos[:, 7] = 0.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert extras["reset_env_ids"] == [0, 1]
    assert extras["terminated_env_ids"] == [0, 1]
    assert bool(mx.all(extras["final_policy_observations"][:, -3] == 1.0).item())


def test_mac_franka_stack_instance_randomize_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve the instance-randomized stack trajectories for fixed actions."""

    cfg = MacFrankaStackInstanceRandomizeEnvCfg(num_envs=4, seed=89, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFrankaStackInstanceRandomizeEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFrankaStackInstanceRandomizeEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_franka_stack_instance_randomize_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable instance-randomized Franka stack checkpoint."""

    checkpoint_path = tmp_path / "franka_stack_instance_randomize_policy.npz"
    train_cfg = MacFrankaStackInstanceRandomizeTrainCfg(
        env=MacFrankaStackInstanceRandomizeEnvCfg(num_envs=8, seed=97, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_franka_stack_instance_randomize_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Stack-Cube-Instance-Randomize-Franka-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space

    episode_returns = play_franka_stack_instance_randomize_policy(
        str(checkpoint_path),
        env_cfg=MacFrankaStackInstanceRandomizeEnvCfg(num_envs=1, seed=97, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)


def test_train_franka_stack_instance_randomize_resume_uses_checkpoint_hidden_dim(tmp_path: Path):
    """Resuming the instance-randomized stack task should preserve the checkpoint architecture."""

    first_checkpoint = tmp_path / "franka_stack_instance_randomize_initial.npz"
    resumed_checkpoint = tmp_path / "franka_stack_instance_randomize_resumed.npz"

    initial_result = train_franka_stack_instance_randomize_policy(
        MacFrankaStackInstanceRandomizeTrainCfg(
            env=MacFrankaStackInstanceRandomizeEnvCfg(num_envs=8, seed=101, episode_length_s=0.5),
            hidden_dim=32,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            checkpoint_path=str(first_checkpoint),
            eval_interval=1,
        )
    )
    resumed_result = train_franka_stack_instance_randomize_policy(
        MacFrankaStackInstanceRandomizeTrainCfg(
            env=MacFrankaStackInstanceRandomizeEnvCfg(num_envs=8, seed=101, episode_length_s=0.5),
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
