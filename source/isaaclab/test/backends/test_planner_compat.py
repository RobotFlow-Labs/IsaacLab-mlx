# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the planner compatibility helpers on the MLX/mac path."""

from __future__ import annotations

import pytest

from isaaclab.backends import (
    JointMotionPlanRequest,
    PlannerWorldObstacle,
    PlannerWorldState,
    create_planner_backend,
    interpolate_joint_motion,
    interpolate_joint_motion_batch,
    resolve_runtime_selection,
)


def test_interpolate_joint_motion_returns_inclusive_endpoints():
    """The planner compatibility layer should emit deterministic inclusive waypoints."""
    request = JointMotionPlanRequest(
        joint_names=("joint_1", "joint_2"),
        start_positions=(0.0, -1.0),
        goal_positions=(1.0, 1.0),
        num_waypoints=5,
        duration_s=2.0,
    )

    plan = interpolate_joint_motion(request, planner_backend="mac-planners")

    assert plan.duration_s == 2.0
    assert plan.waypoints[0] == (0.0, -1.0)
    assert plan.waypoints[2] == pytest.approx((0.5, 0.0))
    assert plan.waypoints[-1] == (1.0, 1.0)
    assert plan.waypoint_times_s == pytest.approx((0.0, 0.5, 1.0, 1.5, 2.0))


def test_mac_planner_backend_tracks_world_state_and_batch_plans():
    """The mac planner backend should expose a usable joint-space planning seam."""
    backend = create_planner_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
    world_state = PlannerWorldState(
        obstacles=(
            PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),
            PlannerWorldObstacle("goal-sphere", kind="sphere", center=(0.5, -0.2, 0.3), radius=0.12),
            PlannerWorldObstacle(
                "wrist-camera",
                kind="mesh",
                center=(0.0, 0.0, 0.08),
                size=(1.0, 1.0, 1.0),
                mesh_resource="package://robotflow/meshes/wrist_camera.usd",
                attached_to="panda_hand",
                touch_links=("panda_hand", "panda_leftfinger", "panda_rightfinger"),
            ),
        )
    )
    backend.update_world_state(world_state)

    requests = (
        JointMotionPlanRequest(
            joint_names=("joint_1", "joint_2"),
            start_positions=(0.0, 0.0),
            goal_positions=(0.8, -0.4),
            num_waypoints=4,
        ),
        JointMotionPlanRequest(
            joint_names=("joint_1", "joint_2"),
            start_positions=(0.8, -0.4),
            goal_positions=(0.1, 0.2),
            num_waypoints=3,
        ),
    )

    batch = backend.plan_joint_motion_batch(requests)

    assert len(batch) == 2
    assert batch[0].world_state["obstacle_count"] == 3
    assert batch[0].world_state["attached_obstacle_count"] == 1
    assert batch[0].world_state["obstacle_type_counts"] == {"box": 1, "mesh": 1, "sphere": 1}
    assert batch[0].waypoints[-1] == pytest.approx((0.8, -0.4))
    assert batch[1].waypoints[-1] == pytest.approx((0.1, 0.2))
    assert batch[0].waypoint_times_s[-1] == pytest.approx(1.0)
    assert backend.state_dict()["world_state"]["obstacle_count"] == 3
    assert backend.state_dict()["world_state"]["attached_obstacle_count"] == 1


def test_interpolate_joint_motion_batch_validates_request_shape():
    """Batched interpolation should preserve request ordering and shape."""
    requests = (
        JointMotionPlanRequest(
            joint_names=("joint_1",),
            start_positions=(0.0,),
            goal_positions=(1.0,),
            num_waypoints=2,
        ),
        JointMotionPlanRequest(
            joint_names=("joint_1",),
            start_positions=(1.0,),
            goal_positions=(0.0,),
            num_waypoints=2,
        ),
    )

    batch = interpolate_joint_motion_batch(requests, planner_backend="mac-planners")

    assert [plan.waypoints[-1][0] for plan in batch] == [1.0, 0.0]


def test_planner_world_obstacle_validates_kind_specific_fields():
    """World obstacles should reject incomplete kind-specific payloads."""

    with pytest.raises(ValueError, match="positive radius"):
        PlannerWorldObstacle("goal-sphere", kind="sphere", center=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="mesh_resource"):
        PlannerWorldObstacle("tool", kind="mesh", center=(0.0, 0.0, 0.0))
