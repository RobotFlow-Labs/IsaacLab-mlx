# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the plain ROS 2 compatibility bridge."""

from __future__ import annotations

from pathlib import Path

from isaaclab.backends import Ros2JsonlBridge, Ros2MessageEnvelope, Ros2ProcessBridge, ros2_cli_available


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
