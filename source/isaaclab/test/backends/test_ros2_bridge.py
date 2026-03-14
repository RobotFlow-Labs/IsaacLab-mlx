# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the plain ROS 2 compatibility bridge."""

from __future__ import annotations

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
    ros2_cli_available,
)


def test_ros2_message_envelope_normalizes_payload():
    """The message envelope should normalize tuple-heavy payloads into JSON-safe structures."""
    envelope = Ros2MessageEnvelope(
        topic="/joint_states",
        msg_type="sensor_msgs/msg/JointState",
        payload={
            "name": ("joint_1", "joint_2"),
            "position": (0.1, -0.2),
            "effort": [0.0, 0.0],
        },
        frame_id="base_link",
        stamp_ns=123,
    )

    normalized = envelope.normalized_payload()

    assert normalized["name"] == ["joint_1", "joint_2"]
    assert normalized["position"] == [0.1, -0.2]
    assert envelope.state_dict()["frame_id"] == "base_link"


def test_ros2_process_bridge_builds_pub_and_echo_commands():
    """The process bridge should build CLI-safe publish and echo commands without requiring ROS imports."""
    bridge = Ros2ProcessBridge()
    envelope = Ros2MessageEnvelope(
        topic="/cmd_vel",
        msg_type="geometry_msgs/msg/Twist",
        payload={"linear": {"x": 0.1, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.3}},
    )

    pub_command = bridge.build_topic_pub_command(envelope)
    echo_command = bridge.build_topic_echo_command("/cmd_vel")

    assert pub_command[:4] == ["ros2", "topic", "pub", "--once"]
    assert pub_command[4] == "/cmd_vel"
    assert pub_command[5] == "geometry_msgs/msg/Twist"
    assert echo_command == ["ros2", "topic", "echo", "/cmd_vel", "--once"]
    assert isinstance(ros2_cli_available(), bool)


def test_ros2_jsonl_bridge_round_trips_messages(tmp_path: Path):
    """The JSONL bridge should persist plain ROS-like envelopes without a ROS install."""
    messages = [
        Ros2MessageEnvelope(
            topic="/clock",
            msg_type="rosgraph_msgs/msg/Clock",
            payload={"clock": {"sec": 1, "nanosec": 2}},
        ),
        Ros2MessageEnvelope(
            topic="/status",
            msg_type="std_msgs/msg/String",
            payload={"data": "mlx-mac-sim"},
        ),
    ]

    output_path = Ros2JsonlBridge.write_messages(tmp_path / "ros2-bridge.jsonl", messages)
    restored = Ros2JsonlBridge.read_messages(output_path)

    assert output_path.exists()
    assert [message.topic for message in restored] == ["/clock", "/status"]
    assert restored[1].payload["data"] == "mlx-mac-sim"


def test_joint_motion_plan_to_ros_envelope_builds_joint_trajectory_payload():
    """Planner plans should export to ROS-friendly joint trajectory envelopes."""
    plan = interpolate_joint_motion(
        JointMotionPlanRequest(
            joint_names=("joint_1", "joint_2"),
            start_positions=(0.0, -0.2),
            goal_positions=(0.6, 0.4),
            num_waypoints=4,
            duration_s=1.2,
        ),
        planner_backend="mac-planners",
    )

    envelope = joint_motion_plan_to_ros_envelope(plan, topic="/planner/joint_trajectory")

    assert envelope.topic == "/planner/joint_trajectory"
    assert envelope.msg_type == "trajectory_msgs/msg/JointTrajectory"
    assert envelope.payload["joint_names"] == ["joint_1", "joint_2"]
    assert len(envelope.payload["points"]) == 4
    assert envelope.payload["points"][-1]["time_from_start"] == {"sec": 1, "nanosec": 200000000}


def test_planner_world_state_to_ros_envelope_preserves_richer_obstacle_metadata():
    """Planner world-state exports should preserve obstacle kind and attachment metadata."""
    world_state = PlannerWorldState(
        frame_id="panda_link0",
        obstacles=(
            PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),
            PlannerWorldObstacle("goal", kind="sphere", center=(0.4, -0.1, 0.2), radius=0.08),
            PlannerWorldObstacle(
                "wrist_camera",
                kind="mesh",
                center=(0.0, 0.0, 0.05),
                size=(1.0, 1.0, 1.0),
                mesh_resource="package://robotflow/meshes/wrist_camera.usd",
                attached_to="panda_hand",
            ),
        ),
    )

    envelope = planner_world_state_to_ros_envelope(world_state)

    assert envelope.msg_type == "robotflow_msgs/msg/PlannerWorldState"
    assert envelope.frame_id == "panda_link0"
    assert envelope.payload["obstacle_count"] == 3
    assert envelope.payload["attached_obstacle_count"] == 1
    assert envelope.payload["obstacle_type_counts"] == {"box": 1, "mesh": 1, "sphere": 1}
