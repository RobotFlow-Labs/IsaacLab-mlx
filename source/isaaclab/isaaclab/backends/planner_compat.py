# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner compatibility helpers for the mac-native MLX path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Any


PlannerObstacleKind = Literal["box", "sphere", "capsule", "mesh"]


@dataclass(frozen=True)
class PlannerWorldObstacle:
    """Serializable obstacle primitive for planner world-state updates."""

    name: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] | None = None
    kind: PlannerObstacleKind = "box"
    radius: float | None = None
    length: float | None = None
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame_id: str = "world"
    attached_to: str | None = None
    touch_links: tuple[str, ...] = ()
    mesh_resource: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.center) != 3:
            raise ValueError("center must have exactly three coordinates")
        if len(self.quaternion_wxyz) != 4:
            raise ValueError("quaternion_wxyz must have exactly four values")
        if self.kind == "box":
            if self.size is None:
                raise ValueError("box obstacles require size")
            if len(self.size) != 3 or any(value <= 0.0 for value in self.size):
                raise ValueError("box obstacle size must contain three positive values")
        elif self.kind == "sphere":
            if self.radius is None or self.radius <= 0.0:
                raise ValueError("sphere obstacles require a positive radius")
        elif self.kind == "capsule":
            if self.radius is None or self.radius <= 0.0:
                raise ValueError("capsule obstacles require a positive radius")
            if self.length is None or self.length <= 0.0:
                raise ValueError("capsule obstacles require a positive length")
        elif self.kind == "mesh":
            if not self.mesh_resource:
                raise ValueError("mesh obstacles require mesh_resource")
            if self.size is not None and (len(self.size) != 3 or any(value <= 0.0 for value in self.size)):
                raise ValueError("mesh obstacle size must contain three positive scale values when provided")

    def dimensions(self) -> list[float]:
        if self.kind == "box":
            assert self.size is not None
            return [float(value) for value in self.size]
        if self.kind == "sphere":
            assert self.radius is not None
            return [float(self.radius)]
        if self.kind == "capsule":
            assert self.radius is not None and self.length is not None
            return [float(self.radius), float(self.length)]
        if self.size is None:
            return [1.0, 1.0, 1.0]
        return [float(value) for value in self.size]

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "frame_id": self.frame_id,
            "center": list(self.center),
            "dimensions": self.dimensions(),
            "size": None if self.size is None else list(self.size),
            "radius": self.radius,
            "length": self.length,
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "attached_to": self.attached_to,
            "touch_links": list(self.touch_links),
            "mesh_resource": self.mesh_resource,
            "metadata": {} if self.metadata is None else dict(self.metadata),
        }


@dataclass(frozen=True)
class PlannerWorldState:
    """Compact world-state payload for planner compatibility backends."""

    frame_id: str = "world"
    obstacles: tuple[PlannerWorldObstacle, ...] = ()

    def state_dict(self) -> dict[str, Any]:
        obstacle_type_counts: dict[str, int] = {}
        attached_obstacle_count = 0
        obstacle_frames: set[str] = set()
        for obstacle in self.obstacles:
            obstacle_type_counts[obstacle.kind] = obstacle_type_counts.get(obstacle.kind, 0) + 1
            if obstacle.attached_to is not None:
                attached_obstacle_count += 1
            obstacle_frames.add(obstacle.frame_id)
        return {
            "frame_id": self.frame_id,
            "obstacle_count": len(self.obstacles),
            "attached_obstacle_count": attached_obstacle_count,
            "obstacle_type_counts": obstacle_type_counts,
            "obstacle_frames": sorted(obstacle_frames),
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
    waypoint_times_s: tuple[float, ...]
    duration_s: float
    planner_backend: str
    world_state: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "waypoint_count": len(self.waypoints),
            "waypoints": [list(waypoint) for waypoint in self.waypoints],
            "waypoint_times_s": list(self.waypoint_times_s),
            "duration_s": self.duration_s,
            "planner_backend": self.planner_backend,
            "first_waypoint": list(self.waypoints[0]),
            "last_waypoint": list(self.waypoints[-1]),
            "first_waypoint_time_s": self.waypoint_times_s[0],
            "last_waypoint_time_s": self.waypoint_times_s[-1],
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
    waypoint_times_s: list[float] = []
    span = request.num_waypoints - 1
    for index in range(request.num_waypoints):
        alpha = index / span
        waypoint_times_s.append(float(alpha * request.duration_s))
        waypoint_rows.append(
            tuple(
                float(start + alpha * (goal - start))
                for start, goal in zip(request.start_positions, request.goal_positions, strict=True)
            )
        )

    return JointMotionPlan(
        joint_names=request.joint_names,
        waypoints=tuple(waypoint_rows),
        waypoint_times_s=tuple(waypoint_times_s),
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
