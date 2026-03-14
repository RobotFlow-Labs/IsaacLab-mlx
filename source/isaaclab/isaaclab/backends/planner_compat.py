# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner compatibility helpers for the mac-native MLX path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlannerWorldObstacle:
    """Serializable obstacle primitive for planner world-state updates."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "center": list(self.center),
            "size": list(self.size),
        }


@dataclass(frozen=True)
class PlannerWorldState:
    """Compact world-state payload for planner compatibility backends."""

    frame_id: str = "world"
    obstacles: tuple[PlannerWorldObstacle, ...] = ()

    def state_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "obstacle_count": len(self.obstacles),
            "obstacles": [obstacle.state_dict() for obstacle in self.obstacles],
        }


@dataclass(frozen=True)
class JointMotionPlanRequest:
    """Request payload for joint-space planning compatibility backends."""

    joint_names: tuple[str, ...]
    start_positions: tuple[float, ...]
    goal_positions: tuple[float, ...]
    num_waypoints: int = 32
    duration_s: float = 1.0

    def __post_init__(self) -> None:
        if len(self.joint_names) == 0:
            raise ValueError("joint_names must not be empty")
        if len(self.joint_names) != len(self.start_positions) or len(self.joint_names) != len(self.goal_positions):
            raise ValueError("joint_names, start_positions, and goal_positions must have the same length")
        if self.num_waypoints < 2:
            raise ValueError("num_waypoints must be >= 2")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be > 0")

    def state_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "start_positions": list(self.start_positions),
            "goal_positions": list(self.goal_positions),
            "num_waypoints": self.num_waypoints,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class JointMotionPlan:
    """Serializable joint-space trajectory emitted by planner compatibility backends."""

    joint_names: tuple[str, ...]
    waypoints: tuple[tuple[float, ...], ...]
    duration_s: float
    planner_backend: str
    world_state: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "waypoint_count": len(self.waypoints),
            "duration_s": self.duration_s,
            "planner_backend": self.planner_backend,
            "first_waypoint": list(self.waypoints[0]),
            "last_waypoint": list(self.waypoints[-1]),
            "world_state": self.world_state,
        }


def interpolate_joint_motion(
    request: JointMotionPlanRequest,
    *,
    planner_backend: str,
    world_state: PlannerWorldState | None = None,
) -> JointMotionPlan:
    """Create a deterministic joint-space interpolation plan."""

    waypoint_rows: list[tuple[float, ...]] = []
    span = request.num_waypoints - 1
    for index in range(request.num_waypoints):
        alpha = index / span
        waypoint_rows.append(
            tuple(
                float(start + alpha * (goal - start))
                for start, goal in zip(request.start_positions, request.goal_positions, strict=True)
            )
        )

    return JointMotionPlan(
        joint_names=request.joint_names,
        waypoints=tuple(waypoint_rows),
        duration_s=request.duration_s,
        planner_backend=planner_backend,
        world_state=None if world_state is None else world_state.state_dict(),
    )


def interpolate_joint_motion_batch(
    requests: tuple[JointMotionPlanRequest, ...],
    *,
    planner_backend: str,
    world_state: PlannerWorldState | None = None,
) -> tuple[JointMotionPlan, ...]:
    """Create deterministic joint-space interpolation plans for a batch of requests."""

    return tuple(
        interpolate_joint_motion(request, planner_backend=planner_backend, world_state=world_state) for request in requests
    )
