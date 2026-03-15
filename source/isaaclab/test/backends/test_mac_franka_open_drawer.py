# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native Franka open-drawer slice."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFrankaOpenDrawerEnv,
    MacFrankaOpenDrawerEnvCfg,
    MacFrankaOpenDrawerTrainCfg,
    play_franka_open_drawer_policy,
    replay_actions,
    rollout_env,
    train_franka_open_drawer_policy,
)


def test_mac_franka_open_drawer_reset_and_step_shapes():
    """The Franka open-drawer env should expose deterministic IsaacLab-style tensors."""

    cfg = MacFrankaOpenDrawerEnvCfg(num_envs=8, seed=73, episode_length_s=0.5)
    env = MacFrankaOpenDrawerEnv(cfg)

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
    assert env.sim_backend.state_dict()["task"] == "franka-open-drawer"


def test_mac_franka_open_drawer_can_open_drawer_when_handle_is_grasped():
    """Pulling along +X with the gripper closed should open the drawer and terminate successfully."""

    cfg = MacFrankaOpenDrawerEnvCfg(num_envs=2, seed=79, episode_length_s=0.5)
    env = MacFrankaOpenDrawerEnv(cfg)
    env.sim_backend.grasped_handle[:] = True
    env.sim_backend.drawer_opened[:] = False
    env.sim_backend.drawer_open_amount[:] = 0.0
    env.sim_backend.handle_anchor_pos_w[:, 0] = env.sim_backend.ee_pos_w[:, 0] - (cfg.drawer_success_distance + 0.03)
    env.sim_backend.handle_anchor_pos_w[:, 1] = env.sim_backend.ee_pos_w[:, 1]
    env.sim_backend.handle_anchor_pos_w[:, 2] = env.sim_backend.ee_pos_w[:, 2]
    env.sim_backend.handle_pos_w[:] = env.sim_backend.handle_anchor_pos_w
    env.sim_backend.state.joint_pos[:, 7] = 0.04

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = -1.0
    _, _, terminated, _, extras = env.step(actions)

    assert bool(mx.all(terminated).item())
    assert extras["reset_env_ids"] == [0, 1]
    assert extras["terminated_env_ids"] == [0, 1]
    assert bool(mx.all(extras["final_policy_observations"][:, 25] >= cfg.drawer_success_distance).item())
    assert bool(mx.all(extras["final_policy_observations"][:, 27] == 1.0).item())


def test_mac_franka_open_drawer_rollout_replay_is_deterministic():
    """Rollout/replay helpers should preserve Franka open-drawer trajectories for fixed actions."""

    cfg = MacFrankaOpenDrawerEnvCfg(num_envs=4, seed=83, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacFrankaOpenDrawerEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacFrankaOpenDrawerEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_train_and_play_franka_open_drawer_smoke(tmp_path: Path):
    """A tiny MLX train/play loop should produce a reusable Franka open-drawer checkpoint."""

    checkpoint_path = tmp_path / "franka_open_drawer_policy.npz"
    train_cfg = MacFrankaOpenDrawerTrainCfg(
        env=MacFrankaOpenDrawerEnvCfg(num_envs=8, seed=89, episode_length_s=0.5),
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint_path=str(checkpoint_path),
        eval_interval=1,
    )

    result = train_franka_open_drawer_policy(train_cfg)

    assert checkpoint_path.exists()
    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_version"] == 2
    assert metadata["checkpoint_format"] == "isaaclab-mlx-ppo"
    assert metadata["task_id"] == "Isaac-Open-Drawer-Franka-v0"
    assert metadata["policy_distribution"] == "gaussian"
    assert metadata["hidden_dim"] == 32

    episode_returns = play_franka_open_drawer_policy(
        str(checkpoint_path),
        env_cfg=MacFrankaOpenDrawerEnvCfg(num_envs=1, seed=89, episode_length_s=0.5),
        episodes=1,
    )

    assert len(episode_returns) == 1
    assert isinstance(episode_returns[0], float)
