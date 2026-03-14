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
    interpolate_joint_motion,
    joint_motion_plan_to_ros_envelope,
    planner_world_state_to_ros_envelope,
)


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
    planner_plan = interpolate_joint_motion(
        JointMotionPlanRequest(
            joint_names=("joint_1", "joint_2"),
            start_positions=(0.0, -0.2),
            goal_positions=(0.4, 0.3),
            num_waypoints=4,
            duration_s=1.0,
        ),
        planner_backend="mac-planners",
        world_state=planner_world,
    )
    messages.extend(
        (
            planner_world_state_to_ros_envelope(planner_world),
            joint_motion_plan_to_ros_envelope(planner_plan),
        )
    )
    output_path = Ros2JsonlBridge.write_messages(args.output, messages)
    restored = Ros2JsonlBridge.read_messages(output_path)
    bridge = Ros2ProcessBridge()

    summary = {
        "cli_available": bridge.cli_available(),
        "message_count": len(restored),
        "first_topic": restored[0].topic,
        "planner_topic": restored[2].topic,
        "trajectory_topic": restored[3].topic,
        "pub_command": bridge.build_topic_pub_command(restored[1]),
        "echo_command": bridge.build_topic_echo_command(restored[1].topic),
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary["message_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
