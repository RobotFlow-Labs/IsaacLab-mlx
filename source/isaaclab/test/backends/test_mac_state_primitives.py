# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for shared mac-sim state primitives."""

from __future__ import annotations

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim.state_primitives import (  # noqa: E402
    BatchedArticulationState,
    BatchedRootState,
    EnvironmentOriginGrid,
    MacSimSceneState,
)


def test_batched_articulation_state_reset_write_and_effort_targets():
    """Joint-state primitives should centralize reset/write/effort-target handling."""
    state = BatchedArticulationState(num_envs=4, num_joints=3)

    state.reset_envs(
        [1, 3],
        joint_pos=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        joint_vel=0.0,
        joint_effort_target=0.0,
    )
    state.set_effort_target([0.25, -0.25, 0.5, -0.5], joint_ids=[0])
    state.write([[7.0, 8.0, 9.0]], [[0.1, 0.2, 0.3]], env_ids=[1])

    joint_pos, joint_vel = state.read()

    assert mx.allclose(joint_pos[1], mx.array([7.0, 8.0, 9.0], dtype=mx.float32)).item()
    assert mx.allclose(joint_pos[3], mx.array([4.0, 5.0, 6.0], dtype=mx.float32)).item()
    assert mx.allclose(joint_vel[1], mx.array([0.1, 0.2, 0.3], dtype=mx.float32)).item()
    assert mx.allclose(state.joint_effort_target[:, 0], mx.array([0.25, -0.25, 0.5, -0.5], dtype=mx.float32)).item()


def test_root_state_uses_shared_origin_grid_and_io_helpers():
    """Root-state primitives should centralize env origins plus pose/velocity IO."""
    grid = EnvironmentOriginGrid(num_envs=4, spacing=2.0)
    state = BatchedRootState(num_envs=4, origin_grid=grid)

    reset_pos = grid.positions_with_offset([0, 3], (0.0, 0.0, 1.0))
    state.reset_envs([0, 3], root_pos_w=reset_pos)
    state.write_root_pose([[0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0]], env_ids=[0])
    state.write_root_velocity([[0.4, 0.0, -0.2, 0.01, 0.02, 0.03]], env_ids=[0])

    assert state.env_origins.shape == (4, 3)
    assert mx.allclose(state.root_pos_w[0], mx.array([0.1, 0.2, 0.9], dtype=mx.float32)).item()
    assert mx.allclose(state.root_quat_w[0], mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)).item()
    assert mx.allclose(state.root_lin_vel_b[0], mx.array([0.4, 0.0, -0.2], dtype=mx.float32)).item()
    assert mx.allclose(state.root_ang_vel_b[0], mx.array([0.01, 0.02, 0.03], dtype=mx.float32)).item()


def test_mac_sim_scene_state_registers_articulations_and_tracks_step_reset():
    """The shared mac-sim scene substrate should manage batched articulations and scene counters."""
    scene = MacSimSceneState(num_envs=3, physics_dt=0.02)
    origin_grid = EnvironmentOriginGrid(num_envs=3, spacing=1.5)
    articulation = scene.add_articulation(
        "robot",
        num_joints=2,
        with_root_state=True,
        origin_grid=origin_grid,
        default_joint_pos=(0.1, -0.1),
        default_root_pose=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
    )

    articulation.write_joint_state([[1.0, 2.0]], [[0.1, 0.2]], env_ids=[1])
    articulation.set_joint_effort_target([0.5, -0.5, 0.25], joint_ids=[0])
    articulation.write_root_pose([[0.2, 0.3, 0.8, 0.0, 0.0, 0.0, 1.0]], env_ids=[2])
    articulation.write_root_velocity([[0.4, 0.0, -0.1, 0.01, 0.02, 0.03]], env_ids=[2])

    scene.step(render=False, update_fabric=True)
    scene.reset()
    scene_state = scene.state_dict()

    assert scene_state["articulation_count"] == 1
    assert scene_state["step_count"] == 0
    assert scene_state["reset_count"] == 1
    assert scene_state["articulations"]["robot"]["root_state_io"] is True
    joint_pos, joint_vel = articulation.get_joint_state()
    assert mx.allclose(joint_pos[0], mx.array([0.1, -0.1], dtype=mx.float32)).item()
    assert mx.allclose(joint_vel[0], mx.array([0.0, 0.0], dtype=mx.float32)).item()
    assert mx.allclose(articulation.root_state.root_pos_w[0], mx.array([0.0, 0.0, 0.5], dtype=mx.float32)).item()
