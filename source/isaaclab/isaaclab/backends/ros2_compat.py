# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plain ROS 2 process/message compatibility helpers for mac-safe runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .planner_compat import JointMotionPlan, PlannerWorldState


def _normalize_message_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_message_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_message_value(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _normalize_message_value(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _normalize_message_value(value.item())
    return repr(value)


def _seconds_to_ros_time(seconds: float) -> dict[str, int]:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return {"sec": sec, "nanosec": nanosec}


@dataclass(frozen=True)
class Ros2MessageEnvelope:
    """Serializable ROS 2 message envelope for CLI and JSONL transport."""

    topic: str
    msg_type: str
    payload: dict[str, Any]
    frame_id: str | None = None
    stamp_ns: int | None = None

    def normalized_payload(self) -> dict[str, Any]:
        return _normalize_message_value(self.payload)

    def state_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "msg_type": self.msg_type,
            "payload": self.normalized_payload(),
            "frame_id": self.frame_id,
            "stamp_ns": self.stamp_ns,
        }


def planner_world_state_to_ros_envelope(
    world_state: PlannerWorldState,
    *,
    topic: str = "/planner/world_state",
    msg_type: str = "robotflow_msgs/msg/PlannerWorldState",
) -> Ros2MessageEnvelope:
    """Convert a planner world-state payload into a ROS-friendly envelope."""

    return Ros2MessageEnvelope(
        topic=topic,
        msg_type=msg_type,
        payload=world_state.state_dict(),
        frame_id=world_state.frame_id,
    )


def joint_motion_plan_to_ros_envelope(
    plan: JointMotionPlan,
    *,
    topic: str = "/planner/joint_trajectory",
    msg_type: str = "trajectory_msgs/msg/JointTrajectory",
    frame_id: str = "world",
) -> Ros2MessageEnvelope:
    """Convert a joint motion plan into a ROS-friendly joint trajectory envelope."""

    points = []
    for waypoint, time_s in zip(plan.waypoints, plan.waypoint_times_s, strict=True):
        points.append(
            {
                "positions": list(waypoint),
                "time_from_start": _seconds_to_ros_time(float(time_s)),
            }
        )

    return Ros2MessageEnvelope(
        topic=topic,
        msg_type=msg_type,
        payload={
            "header": {"frame_id": frame_id},
            "joint_names": list(plan.joint_names),
            "points": points,
            "planner_backend": plan.planner_backend,
        },
        frame_id=frame_id,
    )


def ros2_cli_available(executable: str = "ros2") -> bool:
    """Return True when the ROS 2 CLI is available on PATH."""

    return shutil.which(executable) is not None


class Ros2ProcessBridge:
    """Build and optionally execute plain ROS 2 CLI commands."""

    def __init__(self, *, ros2_executable: str = "ros2") -> None:
        self.ros2_executable = ros2_executable

    def cli_available(self) -> bool:
        return ros2_cli_available(self.ros2_executable)

    def build_topic_pub_command(self, envelope: Ros2MessageEnvelope, *, once: bool = True) -> list[str]:
        command = [self.ros2_executable, "topic", "pub"]
        if once:
            command.append("--once")
        command.extend(
            [
                envelope.topic,
                envelope.msg_type,
                json.dumps(envelope.normalized_payload(), separators=(",", ":"), sort_keys=True),
            ]
        )
        return command

    def build_topic_echo_command(self, topic: str, *, once: bool = True) -> list[str]:
        command = [self.ros2_executable, "topic", "echo", topic]
        if once:
            command.append("--once")
        return command

    def publish_via_cli(
        self,
        envelope: Ros2MessageEnvelope,
        *,
        once: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if not self.cli_available():
            raise FileNotFoundError(f"ROS 2 CLI '{self.ros2_executable}' is not available on PATH")
        return subprocess.run(
            self.build_topic_pub_command(envelope, once=once),
            check=check,
            text=True,
            capture_output=True,
        )


class Ros2JsonlBridge:
    """Persist ROS-compatible message envelopes without requiring a ROS install."""

    @staticmethod
    def write_messages(path: str | Path, envelopes: list[Ros2MessageEnvelope]) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            for envelope in envelopes:
                stream.write(json.dumps(envelope.state_dict(), sort_keys=True) + "\n")
        return output_path

    @staticmethod
    def read_messages(path: str | Path) -> list[Ros2MessageEnvelope]:
        input_path = Path(path)
        messages: list[Ros2MessageEnvelope] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            messages.append(
                Ros2MessageEnvelope(
                    topic=payload["topic"],
                    msg_type=payload["msg_type"],
                    payload=payload["payload"],
                    frame_id=payload.get("frame_id"),
                    stamp_ns=payload.get("stamp_ns"),
                )
            )
        return messages
