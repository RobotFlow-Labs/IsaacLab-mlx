# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the reduced mac-native UR10 suction stack slices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacUR10LongSuctionStackEnv,
    MacUR10LongSuctionStackEnvCfg,
    MacUR10LongSuctionStackTrainCfg,
    MacUR10ShortSuctionStackEnv,
    MacUR10ShortSuctionStackEnvCfg,
    MacUR10ShortSuctionStackTrainCfg,
    play_ur10_long_suction_stack_policy,
    play_ur10_short_suction_stack_policy,
    replay_actions,
    rollout_env,
    train_ur10_long_suction_stack_policy,
    train_ur10_short_suction_stack_policy,
)


@pytest.mark.parametrize(
    ("env_cls", "cfg_cls", "task_key", "upstream_task_id", "suction_variant"),
    (
        (
            MacUR10LongSuctionStackEnv,
            MacUR10LongSuctionStackEnvCfg,
            "ur10-long-suction-stack",
            "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
            "long",
        ),
        (
            MacUR10ShortSuctionStackEnv,
            MacUR10ShortSuctionStackEnvCfg,
            "ur10-short-suction-stack",
            "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
            "short",
        ),
    ),
)
def test_mac_ur10_stack_reset_and_step_shapes(env_cls, cfg_cls, task_key: str, upstream_task_id: str, suction_variant: str):
    """The reduced UR10 suction stack envs should expose deterministic IsaacLab-style tensors."""

    cfg = cfg_cls(num_envs=8, seed=41, episode_length_s=0.5)
    env = env_cls(cfg)

    obs, extras = env.reset()
    assert extras == {}
    assert obs["policy"].shape == (8, 40)

    joint_pos, joint_vel = env.sim_backend.get_joint_state(None)
    assert joint_pos.shape == (8, 7)
    assert joint_vel.shape == (8, 7)

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    next_obs, reward, terminated, truncated, step_extras = env.step(actions)

    assert next_obs["policy"].shape == (8, 40)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert isinstance(step_extras, dict)
    state = env.sim_backend.state_dict()
    assert state["task"] == task_key
    assert state["upstream_task_id"] == upstream_task_id
    assert state["semantic_contract"] == "reduced-analytic-suction-stack"
    assert state["upstream_alias_semantics_preserved"] is False
    assert state["subsystems"]["stack_logic"] is True
    assert state["subsystems"]["suction_grasp_surrogate"] is True
    assert state["joint_state_shape"] == [8, 7]
    assert cfg.suction_variant == suction_variant


@pytest.mark.parametrize(
    ("env_cls", "cfg_cls"),
    (
        (MacUR10LongSuctionStackEnv, MacUR10LongSuctionStackEnvCfg),
        (MacUR10ShortSuctionStackEnv, MacUR10ShortSuctionStackEnvCfg),
    ),
)
def test_mac_ur10_stack_can_complete_top_stack_release(env_cls, cfg_cls):
    """Opening the suction state over the final stack target should terminate successfully."""

    cfg = cfg_cls(num_envs=2, seed=43, episode_length_s=0.5)
    env = env_cls(cfg)
    stack_offset = mx.array([0.0, 0.0, cfg.stack_offset_z], dtype=mx.float32)
    grasp_offset = mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)

    env.sim_backend.middle_stacked[:] = True
    env.sim_backend.top_stacked[:] = False
    env.sim_backend.middle_grasped[:] = False
    env.sim_backend.top_grasped[:] = True
    env.sim_backend.gripper_joint_pos[:] = 0.0
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + 2.0 * cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.middle_cube_pos_w[:, :] = env.sim_backend.support_cube_pos_w + stack_offset
    env.sim_backend.top_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + grasp_offset

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert extras["reset_env_ids"] == [0, 1]
    assert extras["terminated_env_ids"] == [0, 1]
    assert bool(mx.all(extras["final_policy_observations"][:, 37] == 1.0).item())
    assert bool(mx.all(extras["final_policy_observations"][:, 38] == 1.0).item())
    assert bool(mx.all(extras["final_policy_observations"][:, 39] == 1.0).item())


@pytest.mark.parametrize(
    ("env_cls", "cfg_cls"),
    (
        (MacUR10LongSuctionStackEnv, MacUR10LongSuctionStackEnvCfg),
        (MacUR10ShortSuctionStackEnv, MacUR10ShortSuctionStackEnvCfg),
    ),
)
def test_mac_ur10_stack_rollout_replay_is_deterministic(env_cls, cfg_cls):
    """Rollout/replay helpers should preserve UR10 suction stack traces for fixed actions."""

    cfg = cfg_cls(num_envs=4, seed=47, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = env_cls(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = env_cls(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


@pytest.mark.parametrize(
    ("train_cfg_cls", "env_cfg_cls", "train_fn", "play_fn", "task_id", "checkpoint_stem"),
    (
        (
            MacUR10LongSuctionStackTrainCfg,
            MacUR10LongSuctionStackEnvCfg,
            train_ur10_long_suction_stack_policy,
            play_ur10_long_suction_stack_policy,
            "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
            "ur10_long_suction_stack",
        ),
        (
            MacUR10ShortSuctionStackTrainCfg,
            MacUR10ShortSuctionStackEnvCfg,
            train_ur10_short_suction_stack_policy,
            play_ur10_short_suction_stack_policy,
            "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
            "ur10_short_suction_stack",
        ),
    ),
)
def test_train_and_play_ur10_stack_smoke(
    tmp_path: Path,
    train_cfg_cls,
    env_cfg_cls,
    train_fn,
    play_fn,
    task_id: str,
    checkpoint_stem: str,
):
    """Tiny MLX train/play loops should produce reusable UR10 suction stack checkpoints."""

    checkpoint_path = tmp_path / f"{checkpoint_stem}_policy.npz"
    train_cfg = train_cfg_cls(
        env=env_cfg_cls(num_envs=8, seed=53, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_fn(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == task_id
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32
    assert metadata["action_space"] == train_cfg.env.action_space
    assert metadata["policy_action_space"] == train_cfg.env.action_space
    assert metadata["observation_space"] == train_cfg.env.observation_space
    assert metadata["semantic_contract"] == "reduced-analytic-suction-stack"
    assert metadata["upstream_alias_semantics_preserved"] is False

    episode_returns = play_fn(
        str(checkpoint_path),
        env_cfg=env_cfg_cls(num_envs=1, seed=53, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)
