# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke the mac planner compatibility seam without requiring Isaac Sim or cuRobo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.backends import (
    JointMotionPlanRequest,
    PlannerWorldObstacle,
    PlannerWorldState,
    create_planner_backend,
    joint_motion_plan_batch_to_ros_envelopes,
    joint_motion_plan_batch_from_ros_envelopes,
    joint_motion_plan_to_ros_envelope,
    planner_world_state_batch_to_ros_envelopes,
    planner_world_state_batch_from_ros_envelopes,
    planner_world_state_to_ros_envelope,
    Ros2ProcessBridge,
    resolve_runtime_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test for the mac planner compatibility backend.")
    parser.add_argument("output", type=Path, help="Path to the output JSON artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    planner = create_planner_backend(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    world_state = planner.update_world_state(
        PlannerWorldState(
            obstacles=(
                PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),
                PlannerWorldObstacle("target_sphere", kind="sphere", center=(0.45, -0.12, 0.26), radius=0.08),
                PlannerWorldObstacle(
                    "wrist_camera",
                    kind="mesh",
                    center=(0.0, 0.0, 0.06),
                    size=(1.0, 1.0, 1.0),
                    mesh_resource="package://robotflow/meshes/wrist_camera.usd",
                    attached_to="panda_hand",
                    touch_links=("panda_hand", "panda_leftfinger", "panda_rightfinger"),
                ),
            ),
        )
    )
    plan = planner.plan_joint_motion(
        JointMotionPlanRequest(
            joint_names=("joint_1", "joint_2", "joint_3"),
            start_positions=(0.0, -0.5, 0.25),
            goal_positions=(0.8, 0.25, -0.1),
            num_waypoints=6,
            duration_s=1.5,
        )
    )
    batch = planner.plan_joint_motion_batch(
        (
            JointMotionPlanRequest(
                joint_names=("joint_1", "joint_2", "joint_3"),
                start_positions=(0.8, 0.25, -0.1),
                goal_positions=(0.15, -0.2, 0.35),
                num_waypoints=5,
                duration_s=1.0,
            ),
            JointMotionPlanRequest(
                joint_names=("joint_1", "joint_2", "joint_3"),
                start_positions=(0.15, -0.2, 0.35),
                goal_positions=(0.45, 0.0, 0.1),
                num_waypoints=4,
                duration_s=0.75,
            ),
        )
    )
    planner_world_batch = (
        world_state,
        PlannerWorldState(
            obstacles=world_state.obstacles
            + (
                PlannerWorldObstacle(
                    "tool_fixture",
                    center=(0.3, 0.1, 0.18),
                    size=(0.12, 0.08, 0.2),
                ),
            ),
        ),
    )
    planner_world_batch_envelopes = planner_world_state_batch_to_ros_envelopes(planner_world_batch)
    planner_world_batch_roundtrip = planner_world_state_batch_from_ros_envelopes(
        tuple(reversed(planner_world_batch_envelopes))
    )
    trajectory_batch_envelopes = joint_motion_plan_batch_to_ros_envelopes(batch)
    trajectory_batch_roundtrip = joint_motion_plan_batch_from_ros_envelopes(tuple(reversed(trajectory_batch_envelopes)))
    ros_bridge = Ros2ProcessBridge()
    planner_batch_publish_transcript = ros_bridge.build_batch_publish_transcript(planner_world_batch_envelopes)
    trajectory_batch_publish_transcript = ros_bridge.build_batch_publish_transcript(trajectory_batch_envelopes)

    payload = {
        "planner": planner.state_dict(),
        "plan": plan.state_dict(),
        "batch_plans": [item.state_dict() for item in batch],
        "planner_ros_envelope": planner_world_state_to_ros_envelope(world_state).state_dict(),
        "trajectory_ros_envelope": joint_motion_plan_to_ros_envelope(plan).state_dict(),
        "planner_ros_batch_envelopes": [item.state_dict() for item in planner_world_batch_envelopes],
        "trajectory_ros_batch_envelopes": [item.state_dict() for item in trajectory_batch_envelopes],
        "planner_ros_batch_roundtrip_ok": [item.state_dict() for item in planner_world_batch_roundtrip]
        == [item.state_dict() for item in planner_world_batch],
        "trajectory_ros_batch_roundtrip_ok": [item.state_dict() for item in trajectory_batch_roundtrip]
        == [item.state_dict() for item in batch],
        "planner_ros_batch_pub_commands": ros_bridge.build_topic_pub_batch_commands(planner_world_batch_envelopes),
        "trajectory_ros_batch_pub_commands": ros_bridge.build_topic_pub_batch_commands(trajectory_batch_envelopes),
        "planner_ros_batch_publish_transcript": planner_batch_publish_transcript,
        "trajectory_ros_batch_publish_transcript": trajectory_batch_publish_transcript,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["plan"]["waypoint_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
