# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared reporting helpers for MLX/mac benchmark artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
import statistics
from typing import Any

SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hardware_label(platform_payload: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    env_label = os.getenv("MLX_HARDWARE_LABEL")
    if env_label:
        return env_label
    system = str(platform_payload.get("system", "unknown")).lower()
    machine = str(platform_payload.get("machine", "unknown")).lower()
    return f"{system}-{machine}"


def _run_metadata(
    platform_payload: dict[str, Any],
    *,
    hardware_label: str | None = None,
    generated_at: str | None = None,
    git_sha: str | None = None,
    run_id: str | None = None,
    workflow: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _utc_timestamp(),
        "git_sha": git_sha or os.getenv("GITHUB_SHA"),
        "run_id": run_id or os.getenv("GITHUB_RUN_ID"),
        "workflow": workflow or os.getenv("GITHUB_WORKFLOW"),
        "hardware_label": _hardware_label(platform_payload, override=hardware_label),
    }


def _benchmark_kind(benchmark: dict[str, Any]) -> str:
    return "training" if "train_frames_per_s" in benchmark else "rollout"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _max_entry(values: list[tuple[str, float]]) -> dict[str, Any] | None:
    if not values:
        return None
    task, value = max(values, key=lambda item: item[1])
    return {"task": task, "value": float(value)}


def _task_runtime_excerpt(benchmark: dict[str, Any]) -> dict[str, Any]:
    runtime = benchmark.get("runtime", {})
    return {
        "compute_backend": runtime.get("compute_backend"),
        "kernel_backend": runtime.get("kernel_backend"),
        "sim_backend": runtime.get("sim_backend"),
    }


def _task_contract_excerpt(benchmark: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for field in (
        "observation_dim",
        "action_dim",
        "image_shape",
        "camera_mode",
        "sensor_scan_dim",
        "cart_observation_dim",
        "pendulum_observation_dim",
        "thrust_action",
        "completed_episodes",
        "mean_recent_return",
        "checkpoint_path",
        "metadata_path",
    ):
        if field in benchmark:
            contract[field] = benchmark[field]

    diagnostics = benchmark.get("diagnostics", {})
    sim_backend = diagnostics.get("sim_backend", {})
    subsystems = sim_backend.get("subsystems", {})
    if "hotpath" in subsystems:
        contract["hotpath"] = subsystems["hotpath"]
    sensor = diagnostics.get("sensor", {})
    if "implementation" in sensor:
        contract["sensor_implementation"] = sensor["implementation"]

    if "output_signature" in benchmark:
        contract["output_signature"] = benchmark["output_signature"]
    return contract


def build_benchmark_dashboard(
    results: dict[str, Any],
    *,
    hardware_label: str | None = None,
    generated_at: str | None = None,
    git_sha: str | None = None,
    run_id: str | None = None,
    workflow: str | None = None,
) -> dict[str, Any]:
    """Build a stable benchmark/training dashboard payload."""

    benchmarks = list(results.get("benchmarks", []))
    rollout = [benchmark for benchmark in benchmarks if _benchmark_kind(benchmark) == "rollout"]
    training = [benchmark for benchmark in benchmarks if _benchmark_kind(benchmark) == "training"]

    rollout_throughput = [(benchmark["task"], float(benchmark["env_steps_per_s"])) for benchmark in rollout]
    training_throughput = [(benchmark["task"], float(benchmark["train_frames_per_s"])) for benchmark in training]

    task_entries = []
    for benchmark in benchmarks:
        entry = {
            "task": benchmark["task"],
            "kind": _benchmark_kind(benchmark),
            "cpu_fallback": benchmark["cpu_fallback"],
            "runtime": _task_runtime_excerpt(benchmark),
            "contract": _task_contract_excerpt(benchmark),
        }
        if entry["kind"] == "rollout":
            entry["env_steps_per_s"] = float(benchmark["env_steps_per_s"])
            entry["mean_step_ms"] = float(benchmark["mean_step_ms"])
        else:
            entry["train_frames_per_s"] = float(benchmark["train_frames_per_s"])
            entry["elapsed_s"] = float(benchmark["elapsed_s"])
        task_entries.append(entry)

    return {
        **_run_metadata(
            results.get("platform", {}),
            hardware_label=hardware_label,
            generated_at=generated_at,
            git_sha=git_sha,
            run_id=run_id,
            workflow=workflow,
        ),
        "platform": results.get("platform", {}),
        "parameters": results.get("parameters", {}),
        "task_group": results.get("task_group"),
        "tasks": task_entries,
        "status": {
            "cpu_fallback_detected": bool(results.get("cpu_fallback_detected")),
            "cpu_fallback_tasks": list(results.get("cpu_fallback_tasks", [])),
        },
        "summary": {
            "task_count": len(benchmarks),
            "rollout_task_count": len(rollout),
            "training_task_count": len(training),
            "median_env_steps_per_s": _median([value for _, value in rollout_throughput]),
            "mean_env_steps_per_s": _mean([value for _, value in rollout_throughput]),
            "fastest_rollout": _max_entry(rollout_throughput),
            "median_train_frames_per_s": _median([value for _, value in training_throughput]),
            "mean_train_frames_per_s": _mean([value for _, value in training_throughput]),
            "fastest_training": _max_entry(training_throughput),
        },
    }


def build_benchmark_trend(
    results: dict[str, Any],
    *,
    hardware_label: str | None = None,
    generated_at: str | None = None,
    git_sha: str | None = None,
    run_id: str | None = None,
    workflow: str | None = None,
) -> dict[str, Any]:
    """Build a compact benchmark trend payload for archived M-series comparisons."""

    benchmarks = list(results.get("benchmarks", []))
    entries = []
    for benchmark in benchmarks:
        entry = {
            "task": benchmark["task"],
            "kind": _benchmark_kind(benchmark),
            "cpu_fallback": bool(benchmark["cpu_fallback"]["detected"]),
            "runtime": _task_runtime_excerpt(benchmark),
        }
        if entry["kind"] == "rollout":
            entry["env_steps_per_s"] = float(benchmark["env_steps_per_s"])
            entry["mean_step_ms"] = float(benchmark["mean_step_ms"])
        else:
            entry["train_frames_per_s"] = float(benchmark["train_frames_per_s"])
            entry["mean_recent_return"] = float(benchmark["mean_recent_return"])
            entry["completed_episodes"] = int(benchmark["completed_episodes"])
        entries.append(entry)

    return {
        **_run_metadata(
            results.get("platform", {}),
            hardware_label=hardware_label,
            generated_at=generated_at,
            git_sha=git_sha,
            run_id=run_id,
            workflow=workflow,
        ),
        "platform": results.get("platform", {}),
        "parameters": results.get("parameters", {}),
        "task_group": results.get("task_group"),
        "summary": {
            "task_count": len(entries),
            "cpu_fallback_detected": bool(results.get("cpu_fallback_detected")),
            "cpu_fallback_tasks": list(results.get("cpu_fallback_tasks", [])),
        },
        "tasks": entries,
    }


def build_semantic_drift_snapshot(
    results: dict[str, Any],
    *,
    hardware_label: str | None = None,
    generated_at: str | None = None,
    git_sha: str | None = None,
    run_id: str | None = None,
    workflow: str | None = None,
) -> dict[str, Any]:
    """Build a compact semantic contract snapshot for nightly drift comparisons."""

    tasks: dict[str, Any] = {}
    for benchmark in results.get("benchmarks", []):
        if _benchmark_kind(benchmark) != "rollout":
            continue
        task_name = benchmark["task"]
        tasks[task_name] = {
            "runtime": _task_runtime_excerpt(benchmark),
            "cpu_fallback": bool(benchmark["cpu_fallback"]["detected"]),
            "contract": _task_contract_excerpt(benchmark),
        }

    return {
        **_run_metadata(
            results.get("platform", {}),
            hardware_label=hardware_label,
            generated_at=generated_at,
            git_sha=git_sha,
            run_id=run_id,
            workflow=workflow,
        ),
        "platform": results.get("platform", {}),
        "parameters": results.get("parameters", {}),
        "task_group": results.get("task_group"),
        "task_count": len(tasks),
        "tasks": tasks,
    }


def compare_semantic_drift(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
) -> dict[str, Any]:
    """Compare a semantic contract snapshot with a committed baseline."""

    failures: list[dict[str, Any]] = []

    def record_failure(path: str, expected: Any, actual: Any, reason: str) -> None:
        failures.append(
            {
                "path": path,
                "expected": expected,
                "actual": actual,
                "reason": reason,
            }
        )

    def compare_values(path: str, expected: Any, actual: Any) -> None:
        if isinstance(expected, bool) or isinstance(actual, bool):
            if expected is not actual:
                record_failure(path, expected, actual, "bool_mismatch")
            return

        if isinstance(expected, dict) and isinstance(actual, dict):
            expected_keys = set(expected)
            actual_keys = set(actual)
            if expected_keys != actual_keys:
                record_failure(path, sorted(expected_keys), sorted(actual_keys), "key_mismatch")
                return
            for key in sorted(expected_keys):
                compare_values(f"{path}.{key}" if path else key, expected[key], actual[key])
            return

        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(expected) != len(actual):
                record_failure(path, len(expected), len(actual), "length_mismatch")
                return
            for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
                compare_values(f"{path}[{index}]", expected_item, actual_item)
            return

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if not math.isclose(float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol):
                record_failure(path, expected, actual, "numeric_mismatch")
            return

        if expected != actual:
            record_failure(path, expected, actual, "value_mismatch")

    compare_values("parameters", baseline.get("parameters", {}), current.get("parameters", {}))
    compare_values("tasks", baseline.get("tasks", {}), current.get("tasks", {}))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "baseline_task_count": len(baseline.get("tasks", {})),
        "current_task_count": len(current.get("tasks", {})),
        "passed": not failures,
        "failures": failures,
    }
