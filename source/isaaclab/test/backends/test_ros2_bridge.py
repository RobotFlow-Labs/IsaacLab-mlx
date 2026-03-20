# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the plain ROS 2 compatibility bridge."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from isaaclab.backends import (
    JointMotionPlanRequest,
    PlannerWorldObstacle,
    PlannerWorldState,
    Ros2JsonlBridge,
    Ros2MessageEnvelope,
    Ros2ProcessBridge,
    interpolate_joint_motion,
    joint_motion_plan_batch_from_ros_envelopes,
    joint_motion_plan_batch_to_ros_envelopes,
    joint_motion_plan_from_ros_envelope,
    joint_motion_plan_to_ros_envelope,
    planner_world_state_batch_from_ros_envelopes,
    planner_world_state_batch_to_ros_envelopes,
    planner_world_state_from_ros_envelope,
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
    assert envelope.state_dict()["batch_index"] is None


def test_ros2_message_envelope_preserves_batch_index_in_state_dict():
    """Batch envelopes should keep a typed batch index through serialization."""
    envelope = Ros2MessageEnvelope(
        topic="/planner/world_state/3",
        msg_type="robotflow_msgs/msg/PlannerWorldState",
        payload={"frame_id": "world", "batch_index": 3},
        batch_index=3,
    )

    assert envelope.state_dict()["batch_index"] == 3


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


def test_ros2_process_bridge_builds_batch_pub_commands_in_batch_index_order():
    """Batch publish helpers should honor typed batch indices instead of input order."""
    bridge = Ros2ProcessBridge()
    envelopes = [
        Ros2MessageEnvelope(
            topic="/planner/world_state/1",
            msg_type="robotflow_msgs/msg/PlannerWorldState",
            payload={"frame_id": "world", "batch_index": 1},
            batch_index=1,
        ),
        Ros2MessageEnvelope(
            topic="/planner/world_state/0",
            msg_type="robotflow_msgs/msg/PlannerWorldState",
            payload={"frame_id": "world", "batch_index": 0},
            batch_index=0,
        ),
    ]

    commands = bridge.build_topic_pub_batch_commands(tuple(envelopes))

    assert [command[4] for command in commands] == ["/planner/world_state/0", "/planner/world_state/1"]
    assert [json.loads(command[6])["batch_index"] for command in commands] == [0, 1]


def test_ros2_process_bridge_publishes_batch_in_batch_index_order(monkeypatch):
    """Batch publishing should execute envelopes in typed batch order."""
    bridge = Ros2ProcessBridge()
    envelopes = [
        Ros2MessageEnvelope(
            topic="/planner/joint_trajectory/1",
            msg_type="trajectory_msgs/msg/JointTrajectory",
            payload={"joint_names": ["joint_1"], "batch_index": 1},
            batch_index=1,
        ),
        Ros2MessageEnvelope(
            topic="/planner/joint_trajectory/0",
            msg_type="trajectory_msgs/msg/JointTrajectory",
            payload={"joint_names": ["joint_1"], "batch_index": 0},
            batch_index=0,
        ),
    ]
    published_topics: list[str] = []

    def _publish_via_cli(envelope, *, once=True, check=True):
        published_topics.append(envelope.topic)
        return envelope.topic

    monkeypatch.setattr(bridge, "publish_via_cli", _publish_via_cli)

    results = bridge.publish_batch_via_cli(tuple(envelopes))

    assert published_topics == ["/planner/joint_trajectory/0", "/planner/joint_trajectory/1"]
    assert results == ["/planner/joint_trajectory/0", "/planner/joint_trajectory/1"]


def test_ros2_process_bridge_rejects_invalid_batch_envelopes():
    """Malformed batch envelopes should fail explicitly instead of being inferred."""
    bridge = Ros2ProcessBridge()

    with pytest.raises(ValueError, match="missing batch_index"):
        bridge.build_topic_pub_batch_commands(
            (
                Ros2MessageEnvelope(
                    topic="/planner/world_state/0",
                    msg_type="robotflow_msgs/msg/PlannerWorldState",
                    payload={"frame_id": "world"},
                ),
            )
        )

    with pytest.raises(ValueError, match="does not belong to batch topic root"):
        bridge.build_topic_pub_batch_commands(
            (
                Ros2MessageEnvelope(
                    topic="/planner/world_state/0",
                    msg_type="robotflow_msgs/msg/PlannerWorldState",
                    payload={"frame_id": "world", "batch_index": 0},
                    batch_index=0,
                ),
                Ros2MessageEnvelope(
                    topic="/joint_trajectory/1",
                    msg_type="trajectory_msgs/msg/JointTrajectory",
                    payload={"joint_names": ["joint_1"], "batch_index": 1},
                    batch_index=1,
                ),
            )
        )

    with pytest.raises(ValueError, match="Duplicate batch_index"):
        bridge.build_topic_pub_batch_commands(
            (
                Ros2MessageEnvelope(
                    topic="/planner/world_state/0",
                    msg_type="robotflow_msgs/msg/PlannerWorldState",
                    payload={"frame_id": "world", "batch_index": 0},
                    batch_index=0,
                ),
                Ros2MessageEnvelope(
                    topic="/planner/world_state/1",
                    msg_type="robotflow_msgs/msg/PlannerWorldState",
                    payload={"frame_id": "world", "batch_index": 0},
                    batch_index=0,
                ),
            )
        )

    with pytest.raises(ValueError, match="does not belong to batch topic root"):
        bridge.publish_batch_via_cli(
            (
                Ros2MessageEnvelope(
                    topic="/planner/world_state/0",
                    msg_type="robotflow_msgs/msg/PlannerWorldState",
                    payload={"frame_id": "world", "batch_index": 0},
                    batch_index=0,
                ),
                Ros2MessageEnvelope(
                    topic="/joint_trajectory/1",
                    msg_type="trajectory_msgs/msg/JointTrajectory",
                    payload={"joint_names": ["joint_1"], "batch_index": 1},
                    batch_index=1,
                ),
            )
        )


def test_ros2_jsonl_bridge_round_trips_messages(tmp_path: Path):
    """The JSONL bridge should persist plain ROS-like envelopes without a ROS install."""
    messages = [
        Ros2MessageEnvelope(
            topic="/clock",
            msg_type="rosgraph_msgs/msg/Clock",
            payload={"clock": {"sec": 1, "nanosec": 2}},
        ),
        Ros2MessageEnvelope(
            topic="/planner/world_state/0",
            msg_type="robotflow_msgs/msg/PlannerWorldState",
            payload={"frame_id": "world", "batch_index": 0},
            batch_index=0,
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
    assert [message.topic for message in restored] == ["/clock", "/planner/world_state/0", "/status"]
    assert restored[1].batch_index == 0
    assert restored[2].payload["data"] == "mlx-mac-sim"


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


def test_planner_world_state_round_trips_from_ros_envelope():
    """Planner world envelopes should reconstruct the original typed world state."""

    world_state = PlannerWorldState(
        frame_id="panda_link0",
        obstacles=(
            PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),
            PlannerWorldObstacle("goal", kind="sphere", center=(0.4, -0.1, 0.2), radius=0.08),
            PlannerWorldObstacle(
                "tool",
                kind="mesh",
                center=(0.0, 0.0, 0.05),
                size=(1.0, 1.0, 1.0),
                mesh_resource="package://robotflow/meshes/tool.usd",
                attached_to="panda_hand",
                touch_links=("panda_hand",),
            ),
        ),
    )

    restored = planner_world_state_from_ros_envelope(planner_world_state_to_ros_envelope(world_state))

    assert restored.state_dict() == world_state.state_dict()


def test_joint_motion_plan_round_trips_from_ros_envelope():
    """Joint trajectory envelopes should reconstruct the original deterministic motion plan."""

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

    restored = joint_motion_plan_from_ros_envelope(joint_motion_plan_to_ros_envelope(plan))

    assert restored.joint_names == plan.joint_names
    for restored_waypoint, planned_waypoint in zip(restored.waypoints, plan.waypoints, strict=True):
        assert restored_waypoint == pytest.approx(planned_waypoint)
    assert restored.waypoint_times_s == pytest.approx(plan.waypoint_times_s)
    assert restored.duration_s == pytest.approx(plan.duration_s)
    assert restored.planner_backend == plan.planner_backend


def test_planner_world_state_batch_round_trips_from_ros_envelopes():
    """Planner world-state batches should round-trip through ROS-friendly envelopes."""

    batch = (
        PlannerWorldState(
            frame_id="panda_link0",
            obstacles=(PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),),
        ),
        PlannerWorldState(
            frame_id="panda_link0",
            obstacles=(
                PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),
                PlannerWorldObstacle("goal", kind="sphere", center=(0.4, -0.1, 0.2), radius=0.08),
            ),
        ),
    )

    envelopes = planner_world_state_batch_to_ros_envelopes(batch)
    assert [envelope.batch_index for envelope in envelopes] == [0, 1]
    restored = planner_world_state_batch_from_ros_envelopes(tuple(reversed(envelopes)))

    assert [item.state_dict() for item in restored] == [item.state_dict() for item in batch]


def test_joint_motion_plan_batch_round_trips_from_ros_envelopes():
    """Joint trajectory batches should round-trip through ROS-friendly envelopes."""

    batch = (
        interpolate_joint_motion(
            JointMotionPlanRequest(
                joint_names=("joint_1", "joint_2"),
                start_positions=(0.0, -0.2),
                goal_positions=(0.6, 0.4),
                num_waypoints=4,
                duration_s=1.2,
            ),
            planner_backend="mac-planners",
        ),
        interpolate_joint_motion(
            JointMotionPlanRequest(
                joint_names=("joint_1", "joint_2"),
                start_positions=(0.6, 0.4),
                goal_positions=(0.1, -0.1),
                num_waypoints=5,
                duration_s=1.5,
            ),
            planner_backend="mac-planners",
        ),
    )

    envelopes = joint_motion_plan_batch_to_ros_envelopes(batch)
    assert [envelope.batch_index for envelope in envelopes] == [0, 1]
    restored = joint_motion_plan_batch_from_ros_envelopes(tuple(reversed(envelopes)))

    for restored_plan, planned in zip(restored, batch, strict=True):
        assert restored_plan.joint_names == planned.joint_names
        assert restored_plan.waypoint_times_s == pytest.approx(planned.waypoint_times_s)
        assert len(restored_plan.waypoints) == len(planned.waypoints)
        for restored_waypoint, planned_waypoint in zip(restored_plan.waypoints, planned.waypoints, strict=True):
            assert restored_waypoint == pytest.approx(planned_waypoint)


def test_ros2_bridge_smoke_uses_planner_backend_round_trip(tmp_path: Path):
    """The smoke script should exercise the real planner backend and emit round-trip summary flags."""

    script = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "ros2_bridge_smoke.py"
    output_path = tmp_path / "ros2-bridge.jsonl"
    summary_path = tmp_path / "ros2-bridge-summary.json"

    subprocess.run(
        [sys.executable, str(script), str(output_path), "--summary-out", str(summary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["message_count"] == 6
    assert summary["planner_backend"] == "mac-planners"
    assert summary["planner_roundtrip_ok"] is True
    assert summary["trajectory_roundtrip_ok"] is True
    assert summary["planner_batch_size"] == 2
    assert summary["trajectory_batch_size"] == 2
    assert summary["message_summary"]["batch_topics"] == {
        "/planner/joint_trajectory": 2,
        "/planner/world_state": 2,
    }
    assert summary["message_summary"]["batch_topic_indices"] == {
        "/planner/joint_trajectory": [0, 1],
        "/planner/world_state": [0, 1],
    }
    assert [command[4] for command in summary["planner_batch_pub_commands"]] == [
        "/planner/world_state/0",
        "/planner/world_state/1",
    ]
    assert [command[4] for command in summary["trajectory_batch_pub_commands"]] == [
        "/planner/joint_trajectory/0",
        "/planner/joint_trajectory/1",
    ]
    assert summary["planner_obstacle_counts"] == [2, 3]
    assert summary["trajectory_waypoint_counts"] == [4, 5]
