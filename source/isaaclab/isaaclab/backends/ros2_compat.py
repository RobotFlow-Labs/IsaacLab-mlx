# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plain ROS 2 process/message compatibility helpers for mac-safe runtimes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .planner_compat import JointMotionPlan, PlannerWorldObstacle, PlannerWorldState


ROS2_MESSAGE_ENVELOPE_SCHEMA_VERSION = 1


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
    batch_index: int | None = None

    def __post_init__(self) -> None:
        if self.batch_index is not None and (not isinstance(self.batch_index, int) or self.batch_index < 0):
            raise ValueError("batch_index must be >= 0 when provided")

    def normalized_payload(self) -> dict[str, Any]:
        return _normalize_message_value(self.payload)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROS2_MESSAGE_ENVELOPE_SCHEMA_VERSION,
            "kind": "ros2_message_envelope",
            "topic": self.topic,
            "msg_type": self.msg_type,
            "payload": self.normalized_payload(),
            "frame_id": self.frame_id,
            "stamp_ns": self.stamp_ns,
            "batch_index": self.batch_index,
        }


@dataclass(frozen=True)
class Ros2BatchPublishRecord:
    """Serializable record for one batch publish attempt through the ROS 2 CLI."""

    batch_index: int
    topic: str
    msg_type: str
    command: tuple[str, ...]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "ros2_batch_publish_record",
            "batch_index": self.batch_index,
            "topic": self.topic,
            "msg_type": self.msg_type,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _batch_index_from_envelope(envelope: Ros2MessageEnvelope) -> int:
    if envelope.batch_index is None:
        payload = envelope.normalized_payload()
        if "batch_index" not in payload:
            raise ValueError(f"ROS envelope '{envelope.topic}' is missing batch_index")
        try:
            batch_index = int(payload["batch_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ROS envelope '{envelope.topic}' has an invalid batch_index") from exc
        if batch_index < 0:
            raise ValueError(f"ROS envelope '{envelope.topic}' has a negative batch_index")
        return batch_index
    return envelope.batch_index


def _topic_root_for_batch(topic: str) -> str:
    if "/" not in topic:
        return topic
    return topic.rsplit("/", 1)[0]


def _sorted_batch_envelopes(
    envelopes: list[Ros2MessageEnvelope] | tuple[Ros2MessageEnvelope, ...],
    *,
    topic_root: str,
) -> list[tuple[int, Ros2MessageEnvelope]]:
    indexed_envelopes: list[tuple[int, Ros2MessageEnvelope]] = []
    seen_indices: set[int] = set()
    for envelope in envelopes:
        if _topic_root_for_batch(envelope.topic) != topic_root:
            raise ValueError(f"ROS envelope '{envelope.topic}' does not belong to batch topic root '{topic_root}'")
        batch_index = _batch_index_from_envelope(envelope)
        if batch_index in seen_indices:
            raise ValueError(f"Duplicate batch_index {batch_index} for batch topic root '{topic_root}'")
        seen_indices.add(batch_index)
        indexed_envelopes.append((batch_index, envelope))
    indexed_envelopes.sort(key=lambda item: item[0])
    return indexed_envelopes


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


def planner_world_state_from_ros_envelope(envelope: Ros2MessageEnvelope) -> PlannerWorldState:
    """Reconstruct a planner world-state from a ROS-friendly envelope payload."""

    payload = envelope.normalized_payload()
    obstacle_payloads = payload.get("obstacles", [])
    obstacles: list[PlannerWorldObstacle] = []
    for obstacle_payload in obstacle_payloads:
        obstacles.append(
            PlannerWorldObstacle(
                name=str(obstacle_payload["name"]),
                center=tuple(float(value) for value in obstacle_payload.get("center", (0.0, 0.0, 0.0))),
                size=(
                    None
                    if obstacle_payload.get("size") is None
                    else tuple(float(value) for value in obstacle_payload["size"])
                ),
                kind=str(obstacle_payload.get("kind", "box")),
                radius=None if obstacle_payload.get("radius") is None else float(obstacle_payload["radius"]),
                length=None if obstacle_payload.get("length") is None else float(obstacle_payload["length"]),
                quaternion_wxyz=tuple(
                    float(value) for value in obstacle_payload.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))
                ),
                frame_id=str(obstacle_payload.get("frame_id", payload.get("frame_id", envelope.frame_id or "world"))),
                attached_to=obstacle_payload.get("attached_to"),
                touch_links=tuple(str(value) for value in obstacle_payload.get("touch_links", ())),
                mesh_resource=obstacle_payload.get("mesh_resource"),
                metadata=(
                    None
                    if obstacle_payload.get("metadata") in (None, {})
                    else dict(obstacle_payload["metadata"])
                ),
            )
        )
    return PlannerWorldState(
        frame_id=str(payload.get("frame_id", envelope.frame_id or "world")),
        obstacles=tuple(obstacles),
    )


def planner_world_state_batch_to_ros_envelopes(
    world_states: list[PlannerWorldState] | tuple[PlannerWorldState, ...],
    *,
    topic_root: str = "/planner/world_state",
    msg_type: str = "robotflow_msgs/msg/PlannerWorldState",
) -> list[Ros2MessageEnvelope]:
    """Convert a planner world-state batch into ROS-friendly envelopes."""

    envelopes: list[Ros2MessageEnvelope] = []
    for index, world_state in enumerate(world_states):
        envelope = planner_world_state_to_ros_envelope(
            world_state,
            topic=f"{topic_root}/{index}",
            msg_type=msg_type,
        )
        payload = envelope.normalized_payload()
        payload["batch_index"] = index
        envelopes.append(
            Ros2MessageEnvelope(
                topic=envelope.topic,
                msg_type=envelope.msg_type,
                payload=payload,
                frame_id=envelope.frame_id,
                stamp_ns=envelope.stamp_ns,
                batch_index=index,
            )
        )
    return envelopes


def planner_world_state_batch_from_ros_envelopes(
    envelopes: list[Ros2MessageEnvelope] | tuple[Ros2MessageEnvelope, ...],
) -> tuple[PlannerWorldState, ...]:
    """Reconstruct a planner world-state batch from ROS-friendly envelopes."""

    if not envelopes:
        return ()
    topic_root = _topic_root_for_batch(envelopes[0].topic)
    indexed_envelopes = _sorted_batch_envelopes(envelopes, topic_root=topic_root)
    return tuple(planner_world_state_from_ros_envelope(envelope) for _, envelope in indexed_envelopes)


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


def joint_motion_plan_from_ros_envelope(envelope: Ros2MessageEnvelope) -> JointMotionPlan:
    """Reconstruct a deterministic joint motion plan from a ROS-friendly envelope payload."""

    payload = envelope.normalized_payload()
    joint_names = tuple(str(name) for name in payload.get("joint_names", ()))
    points = payload.get("points", [])
    waypoints = tuple(tuple(float(value) for value in point.get("positions", ())) for point in points)
    waypoint_times_s = tuple(
        float(point.get("time_from_start", {}).get("sec", 0))
        + float(point.get("time_from_start", {}).get("nanosec", 0)) / 1_000_000_000.0
        for point in points
    )
    duration_s = waypoint_times_s[-1] if waypoint_times_s else 0.0
    return JointMotionPlan(
        joint_names=joint_names,
        waypoints=waypoints,
        waypoint_times_s=waypoint_times_s,
        duration_s=duration_s,
        planner_backend=str(payload.get("planner_backend", "unknown")),
        world_state=None,
    )


def joint_motion_plan_batch_to_ros_envelopes(
    plans: list[JointMotionPlan] | tuple[JointMotionPlan, ...],
    *,
    topic_root: str = "/planner/joint_trajectory",
    msg_type: str = "trajectory_msgs/msg/JointTrajectory",
    frame_id: str = "world",
) -> list[Ros2MessageEnvelope]:
    """Convert a joint motion plan batch into ROS-friendly envelopes."""

    envelopes: list[Ros2MessageEnvelope] = []
    for index, plan in enumerate(plans):
        envelope = joint_motion_plan_to_ros_envelope(
            plan,
            topic=f"{topic_root}/{index}",
            msg_type=msg_type,
            frame_id=frame_id,
        )
        payload = envelope.normalized_payload()
        payload["batch_index"] = index
        envelopes.append(
            Ros2MessageEnvelope(
                topic=envelope.topic,
                msg_type=envelope.msg_type,
                payload=payload,
                frame_id=envelope.frame_id,
                stamp_ns=envelope.stamp_ns,
                batch_index=index,
            )
        )
    return envelopes


def joint_motion_plan_batch_from_ros_envelopes(
    envelopes: list[Ros2MessageEnvelope] | tuple[Ros2MessageEnvelope, ...],
) -> tuple[JointMotionPlan, ...]:
    """Reconstruct a deterministic joint motion plan batch from ROS-friendly envelopes."""

    if not envelopes:
        return ()
    topic_root = _topic_root_for_batch(envelopes[0].topic)
    indexed_envelopes = _sorted_batch_envelopes(envelopes, topic_root=topic_root)
    return tuple(joint_motion_plan_from_ros_envelope(envelope) for _, envelope in indexed_envelopes)


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

    def build_topic_pub_batch_commands(
        self,
        envelopes: list[Ros2MessageEnvelope] | tuple[Ros2MessageEnvelope, ...],
        *,
        once: bool = True,
    ) -> list[list[str]]:
        """Build CLI publish commands for a batch of envelopes in batch-index order."""

        if not envelopes:
            return []
        topic_root = _topic_root_for_batch(envelopes[0].topic)
        indexed_envelopes = _sorted_batch_envelopes(envelopes, topic_root=topic_root)
        return [self.build_topic_pub_command(envelope, once=once) for _, envelope in indexed_envelopes]

    def build_batch_publish_manifest(
        self,
        envelopes: list[Ros2MessageEnvelope] | tuple[Ros2MessageEnvelope, ...],
        *,
        once: bool = True,
    ) -> list[Ros2BatchPublishRecord]:
        """Build a replay-safe batch publish manifest in typed batch-index order."""

        if not envelopes:
            return []
        indexed_envelopes = _sorted_batch_envelopes(envelopes, topic_root=_topic_root_for_batch(envelopes[0].topic))
        return [
            Ros2BatchPublishRecord(
                batch_index=batch_index,
                topic=envelope.topic,
                msg_type=envelope.msg_type,
                command=tuple(self.build_topic_pub_command(envelope, once=once)),
            )
            for batch_index, envelope in indexed_envelopes
        ]

    def publish_batch_via_cli(
        self,
        envelopes: list[Ros2MessageEnvelope] | tuple[Ros2MessageEnvelope, ...],
        *,
        once: bool = True,
        check: bool = True,
    ) -> list[Ros2BatchPublishRecord]:
        """Publish a typed ROS batch sequentially in batch-index order."""

        if not envelopes:
            return []
        indexed_envelopes = _sorted_batch_envelopes(envelopes, topic_root=_topic_root_for_batch(envelopes[0].topic))
        results: list[Ros2BatchPublishRecord] = []
        for batch_index, envelope in indexed_envelopes:
            record = Ros2BatchPublishRecord(
                batch_index=batch_index,
                topic=envelope.topic,
                msg_type=envelope.msg_type,
                command=tuple(self.build_topic_pub_command(envelope, once=once)),
            )
            try:
                completed = self.publish_via_cli(envelope, once=once, check=check)
            except subprocess.CalledProcessError as exc:
                failed_record = replace(
                    record,
                    returncode=exc.returncode,
                    stdout=exc.output or "",
                    stderr=exc.stderr or "",
                )
                results.append(failed_record)
                error = RuntimeError(
                    f"ROS 2 batch publish failed at batch_index={batch_index} topic='{envelope.topic}' "
                    f"returncode={exc.returncode}"
                )
                error.batch_publish_manifest = [item.state_dict() for item in results]
                raise error from exc
            results.append(
                replace(
                    record,
                    returncode=completed.returncode,
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                )
            )
        return results

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
            if int(payload.get("schema_version", ROS2_MESSAGE_ENVELOPE_SCHEMA_VERSION)) != ROS2_MESSAGE_ENVELOPE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported ROS 2 envelope schema version: {payload.get('schema_version')}"
                )
            if payload.get("kind", "ros2_message_envelope") != "ros2_message_envelope":
                raise ValueError(f"Unsupported ROS 2 envelope kind: {payload.get('kind')}")
            messages.append(
                Ros2MessageEnvelope(
                    topic=payload["topic"],
                    msg_type=payload["msg_type"],
                    payload=payload["payload"],
                    frame_id=payload.get("frame_id"),
                    stamp_ns=payload.get("stamp_ns"),
                    batch_index=payload.get("batch_index"),
                )
            )
        return messages

    @staticmethod
    def summarize_messages(messages: list[Ros2MessageEnvelope]) -> dict[str, Any]:
        """Summarize plain ROS-like envelopes for smoke tests and CI."""

        topic_counts: dict[str, int] = {}
        msg_type_counts: dict[str, int] = {}
        batch_topics: dict[str, int] = {}
        batch_topic_indices: dict[str, list[int]] = {}
        for envelope in messages:
            topic_counts[envelope.topic] = topic_counts.get(envelope.topic, 0) + 1
            msg_type_counts[envelope.msg_type] = msg_type_counts.get(envelope.msg_type, 0) + 1
            if envelope.batch_index is not None or "batch_index" in envelope.normalized_payload():
                topic_root = _topic_root_for_batch(envelope.topic)
                batch_topics[topic_root] = batch_topics.get(topic_root, 0) + 1
                batch_topic_indices.setdefault(topic_root, []).append(
                    envelope.batch_index if envelope.batch_index is not None else _batch_index_from_envelope(envelope)
                )
        return {
            "message_count": len(messages),
            "schema_version": ROS2_MESSAGE_ENVELOPE_SCHEMA_VERSION,
            "topic_counts": dict(sorted(topic_counts.items())),
            "msg_type_counts": dict(sorted(msg_type_counts.items())),
            "batch_topics": dict(sorted(batch_topics.items())),
            "batch_topic_indices": {topic: sorted(indices) for topic, indices in sorted(batch_topic_indices.items())},
        }
