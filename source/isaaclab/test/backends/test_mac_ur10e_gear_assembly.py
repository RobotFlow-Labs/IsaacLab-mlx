# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the reduced mac-native UR10e gear-assembly slices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacUR10eGearAssembly2F140Env,
    MacUR10eGearAssembly2F140EnvCfg,
    MacUR10eGearAssembly2F140TrainCfg,
    MacUR10eGearAssembly2F85Env,
    MacUR10eGearAssembly2F85EnvCfg,
    MacUR10eGearAssembly2F85TrainCfg,
    play_ur10e_gear_assembly_2f140_policy,
    play_ur10e_gear_assembly_2f85_policy,
    replay_actions,
    rollout_env,
    train_ur10e_gear_assembly_2f140_policy,
    train_ur10e_gear_assembly_2f85_policy,
)


@pytest.mark.parametrize(
    (
        "env_cls",
        "cfg_cls",
        "task_key",
        "upstream_task_id",
        "gripper_variant",
    ),
    (
        (
            MacUR10eGearAssembly2F140Env,
            MacUR10eGearAssembly2F140EnvCfg,
            "ur10e-gear-assembly-2f140",
            "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
            "2f140",
        ),
        (
            MacUR10eGearAssembly2F85Env,
            MacUR10eGearAssembly2F85EnvCfg,
            "ur10e-gear-assembly-2f85",
            "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
            "2f85",
        ),
    ),
)
def test_mac_ur10e_gear_assembly_reset_and_step_shapes(
    env_cls,
    cfg_cls,
    task_key: str,
    upstream_task_id: str,
    gripper_variant: str,
):
    """The reduced UR10e gear-assembly envs should expose deterministic IsaacLab-style tensors."""

    cfg = cfg_cls(num_envs=8, seed=23, episode_length_s=0.5)
    env = env_cls(cfg)

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
    assert state["task"] == task_key
    assert state["upstream_task_id"] == upstream_task_id
    assert state["semantic_contract"] == "reduced-analytic-assembly"
    assert state["upstream_alias_semantics_preserved"] is False
    assert state["subsystems"]["gear_assembly_logic"] is True
    assert state["subsystems"]["gear_type_randomization"] is True
    assert state["gripper_variant"] == gripper_variant
    assert len(state["gear_type_offsets_x"]) == 3


@pytest.mark.parametrize(
    ("env_cls", "cfg_cls"),
    (
        (MacUR10eGearAssembly2F140Env, MacUR10eGearAssembly2F140EnvCfg),
        (MacUR10eGearAssembly2F85Env, MacUR10eGearAssembly2F85EnvCfg),
    ),
)
def test_mac_ur10e_gear_assembly_rollout_replay_is_deterministic(env_cls, cfg_cls):
    """Rollout/replay helpers should preserve UR10e gear-assembly traces for fixed actions."""

    cfg = cfg_cls(num_envs=4, seed=29, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = env_cls(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = env_cls(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


@pytest.mark.parametrize(
    ("env_cls", "cfg_cls"),
    (
        (MacUR10eGearAssembly2F140Env, MacUR10eGearAssembly2F140EnvCfg),
        (MacUR10eGearAssembly2F85Env, MacUR10eGearAssembly2F85EnvCfg),
    ),
)
def test_mac_ur10e_gear_assembly_policy_observations_track_insertion_progress(env_cls, cfg_cls):
    """The reduced 19D observation should stay Markov by encoding insertion progress in the shaft pose."""

    cfg = cfg_cls(num_envs=2, seed=31, episode_length_s=0.5)
    env = env_cls(cfg)

    before = env._build_policy_observations()
    env.sim_backend.shaft_depth = mx.full((cfg.num_envs,), 0.02, dtype=mx.float32)
    after = env._build_policy_observations()

    assert before.shape == after.shape == (2, 19)
    assert not mx.allclose(before[:, 12:15], after[:, 12:15]).item()
    assert mx.allclose(before[:, 15:19], after[:, 15:19]).item()


@pytest.mark.parametrize(
    ("train_cfg_cls", "env_cfg_cls", "train_fn", "play_fn", "task_id", "checkpoint_stem"),
    (
        (
            MacUR10eGearAssembly2F140TrainCfg,
            MacUR10eGearAssembly2F140EnvCfg,
            train_ur10e_gear_assembly_2f140_policy,
            play_ur10e_gear_assembly_2f140_policy,
            "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
            "ur10e_gear_assembly_2f140",
        ),
        (
            MacUR10eGearAssembly2F85TrainCfg,
            MacUR10eGearAssembly2F85EnvCfg,
            train_ur10e_gear_assembly_2f85_policy,
            play_ur10e_gear_assembly_2f85_policy,
            "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
            "ur10e_gear_assembly_2f85",
        ),
    ),
)
def test_train_and_play_ur10e_gear_assembly_smoke(
    tmp_path: Path,
    train_cfg_cls,
    env_cfg_cls,
    train_fn,
    play_fn,
    task_id: str,
    checkpoint_stem: str,
):
    """Tiny MLX train/play loops should produce reusable UR10e gear-assembly checkpoints."""

    checkpoint_path = tmp_path / f"{checkpoint_stem}_policy.npz"
    train_cfg = train_cfg_cls(
        env=env_cfg_cls(num_envs=8, seed=23, episode_length_s=0.5),
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
    assert metadata["semantic_contract"] == "reduced-analytic-assembly"
    assert metadata["upstream_alias_semantics_preserved"] is False

    episode_returns = play_fn(
        str(checkpoint_path),
        env_cfg=env_cfg_cls(num_envs=1, seed=23, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)
