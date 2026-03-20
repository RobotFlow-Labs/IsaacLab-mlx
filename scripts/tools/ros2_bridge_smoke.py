# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke the plain ROS 2 compatibility bridge without requiring a ROS install."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.backends import (
    JointMotionPlanRequest,
    PlannerWorldObstacle,
    PlannerWorldState,
    Ros2JsonlBridge,
    Ros2MessageEnvelope,
    Ros2ProcessBridge,
    create_planner_backend,
    joint_motion_plan_batch_from_ros_envelopes,
    joint_motion_plan_batch_to_ros_envelopes,
    joint_motion_plan_from_ros_envelope,
    joint_motion_plan_to_ros_envelope,
    planner_world_state_batch_from_ros_envelopes,
    planner_world_state_batch_to_ros_envelopes,
    planner_world_state_from_ros_envelope,
    planner_world_state_to_ros_envelope,
    resolve_runtime_selection,
)


def _plans_equivalent(lhs, rhs, *, tol: float = 1e-9) -> bool:
    if lhs.joint_names != rhs.joint_names or lhs.planner_backend != rhs.planner_backend:
        return False
    if len(lhs.waypoints) != len(rhs.waypoints) or len(lhs.waypoint_times_s) != len(rhs.waypoint_times_s):
        return False
    for lhs_waypoint, rhs_waypoint in zip(lhs.waypoints, rhs.waypoints, strict=True):
        if len(lhs_waypoint) != len(rhs_waypoint):
            return False
        if any(abs(float(left) - float(right)) > tol for left, right in zip(lhs_waypoint, rhs_waypoint, strict=True)):
            return False
    return all(abs(float(left) - float(right)) <= tol for left, right in zip(lhs.waypoint_times_s, rhs.waypoint_times_s, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test for the ROS 2 compatibility bridge.")
    parser.add_argument("output", type=Path, help="Path to the JSONL output artifact.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    messages = [
        Ros2MessageEnvelope(
            topic="/clock",
            msg_type="rosgraph_msgs/msg/Clock",
            payload={"clock": {"sec": 1, "nanosec": 2}},
        ),
        Ros2MessageEnvelope(
            topic="/joint_states",
            msg_type="sensor_msgs/msg/JointState",
            payload={
                "name": ("joint_1", "joint_2"),
                "position": (0.1, -0.2),
                "velocity": (0.0, 0.0),
            },
        ),
    ]
    planner_world = PlannerWorldState(
        obstacles=(
            PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),
            PlannerWorldObstacle("goal", kind="sphere", center=(0.4, -0.1, 0.2), radius=0.08),
        )
    )
    planner = create_planner_backend(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    planner_world = planner.update_world_state(planner_world)
    planner_plan = planner.plan_joint_motion_batch(
        (
            JointMotionPlanRequest(
                joint_names=("joint_1", "joint_2"),
                start_positions=(0.0, -0.2),
                goal_positions=(0.4, 0.3),
                num_waypoints=4,
                duration_s=1.0,
            ),
            JointMotionPlanRequest(
                joint_names=("joint_1", "joint_2"),
                start_positions=(0.4, 0.3),
                goal_positions=(0.1, -0.1),
                num_waypoints=5,
                duration_s=1.25,
            ),
        )
    )
    planner_world_batch = (
        planner_world,
        PlannerWorldState(
            obstacles=planner_world.obstacles
            + (
                PlannerWorldObstacle("wrist_camera", kind="mesh", center=(0.0, 0.0, 0.05), size=(1.0, 1.0, 1.0), mesh_resource="package://robotflow/meshes/wrist_camera.usd", attached_to="panda_hand"),
            ),
        ),
    )
    planner_world_batch_envelopes = planner_world_state_batch_to_ros_envelopes(planner_world_batch)
    planner_plan_batch_envelopes = joint_motion_plan_batch_to_ros_envelopes(planner_plan)
    messages.extend(planner_world_batch_envelopes + planner_plan_batch_envelopes)
    output_path = Ros2JsonlBridge.write_messages(args.output, messages)
    restored = Ros2JsonlBridge.read_messages(output_path)
    bridge = Ros2ProcessBridge()
    restored_world_batch = planner_world_state_batch_from_ros_envelopes(
        tuple(reversed([envelope for envelope in restored if envelope.topic.startswith("/planner/world_state/")]))
    )
    restored_plan_batch = joint_motion_plan_batch_from_ros_envelopes(
        tuple(reversed([envelope for envelope in restored if envelope.topic.startswith("/planner/joint_trajectory/")]))
    )
    message_summary = Ros2JsonlBridge.summarize_messages(restored)
    planner_batch_commands = bridge.build_topic_pub_batch_commands(
        tuple(reversed(planner_world_batch_envelopes))
    )
    trajectory_batch_commands = bridge.build_topic_pub_batch_commands(
        tuple(reversed(planner_plan_batch_envelopes))
    )
    planner_batch_publish_manifest = [
        item.state_dict() for item in bridge.build_batch_publish_manifest(tuple(reversed(planner_world_batch_envelopes)))
    ]
    trajectory_batch_publish_manifest = [
        item.state_dict() for item in bridge.build_batch_publish_manifest(tuple(reversed(planner_plan_batch_envelopes)))
    ]

    summary = {
        "cli_available": bridge.cli_available(),
        "message_count": len(restored),
        "first_topic": restored[0].topic,
        "planner_topic": next(envelope.topic for envelope in restored if envelope.topic.startswith("/planner/world_state/")),
        "trajectory_topic": next(
            envelope.topic for envelope in restored if envelope.topic.startswith("/planner/joint_trajectory/")
        ),
        "pub_command": bridge.build_topic_pub_command(restored[1]),
        "echo_command": bridge.build_topic_echo_command(restored[1].topic),
        "planner_backend": planner.state_dict()["backend"],
        "planner_roundtrip_ok": [item.state_dict() for item in restored_world_batch] == [item.state_dict() for item in planner_world_batch],
        "trajectory_roundtrip_ok": all(_plans_equivalent(restored, planned) for restored, planned in zip(restored_plan_batch, planner_plan, strict=True)),
        "planner_obstacle_count": restored_world_batch[0].state_dict()["obstacle_count"],
        "planner_obstacle_counts": [item.state_dict()["obstacle_count"] for item in restored_world_batch],
        "trajectory_waypoint_count": len(restored_plan_batch[0].waypoints),
        "trajectory_waypoint_counts": [len(item.waypoints) for item in restored_plan_batch],
        "planner_batch_size": len(restored_world_batch),
        "trajectory_batch_size": len(restored_plan_batch),
        "planner_batch_pub_commands": planner_batch_commands,
        "trajectory_batch_pub_commands": trajectory_batch_commands,
        "planner_batch_publish_manifest": planner_batch_publish_manifest,
        "trajectory_batch_publish_manifest": trajectory_batch_publish_manifest,
        "message_summary": message_summary,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary["message_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
