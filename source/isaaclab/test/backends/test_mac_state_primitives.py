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
