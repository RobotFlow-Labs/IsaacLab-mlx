# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic drift contract tests for the MLX/mac benchmark suite."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from isaaclab.backends import build_semantic_drift_snapshot, compare_semantic_drift
from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()
from isaaclab.backends.mac_sim.hotpath import (  # noqa: E402
    get_franka_cabinet_hotpath_backend,
    get_franka_hotpath_backend,
    get_franka_lift_hotpath_backend,
    get_franka_stack_hotpath_backend,
    get_franka_stack_rgb_hotpath_backend,
    get_locomotion_hotpath_backend,
    get_ur10e_hotpath_backend,
)


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[4] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_drift_snapshot_covers_rollout_contracts(tmp_path: Path):
    """Semantic snapshots should exclude training tasks and retain rollout/sensor contracts."""
    benchmark_module = _load_module("benchmark_mac_tasks", "scripts/benchmarks/mlx/benchmark_mac_tasks.py")
    expected_franka_hotpath = get_franka_hotpath_backend()
    expected_franka_lift_hotpath = get_franka_lift_hotpath_backend()
    expected_franka_stack_hotpath = get_franka_stack_hotpath_backend()
    expected_franka_stack_rgb_hotpath = get_franka_stack_rgb_hotpath_backend()
    expected_locomotion_hotpath = get_locomotion_hotpath_backend()
    expected_ur10e_hotpath = get_ur10e_hotpath_backend()

    results = benchmark_module.run_benchmarks(
        benchmark_module.resolve_requested_tasks(None, "full"),
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=17,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )
    snapshot = build_semantic_drift_snapshot(results, hardware_label="m5-ultra")

    assert snapshot["hardware_label"] == "m5-ultra"
    assert "train-cartpole" not in snapshot["tasks"]
    assert snapshot["task_count"] == 32
    assert snapshot["tasks"]["cartpole"]["contract"]["observation_dim"] == 4
    assert snapshot["tasks"]["cartpole-rgb-camera"]["contract"]["camera_mode"] == "rgb"
    assert snapshot["tasks"]["cartpole-depth-camera"]["contract"]["image_shape"] == [100, 100, 1]
    assert snapshot["tasks"]["anymal-c-flat"]["contract"]["hotpath"] == expected_locomotion_hotpath
    assert snapshot["tasks"]["anymal-c-flat-height-scan"]["contract"]["sensor_implementation"] == "analytic-terrain-raycast"
    assert snapshot["tasks"]["anymal-c-flat-height-scan"]["contract"]["hotpath"] == expected_locomotion_hotpath
    assert snapshot["tasks"]["franka-reach"]["contract"]["action_dim"] == 7
    assert snapshot["tasks"]["franka-reach"]["contract"]["hotpath"] == expected_franka_hotpath
    assert snapshot["tasks"]["openarm-reach"]["contract"]["action_dim"] == 7
    assert snapshot["tasks"]["openarm-reach"]["contract"]["hotpath"] == expected_franka_hotpath
    assert snapshot["tasks"]["openarm-reach"]["contract"]["output_signature"]["final_target_distance_mean"] >= 0.0
    assert snapshot["tasks"]["openarm-bi-reach"]["contract"]["action_dim"] == 14
    assert snapshot["tasks"]["openarm-bi-reach"]["contract"]["hotpath"] == expected_franka_hotpath
    assert snapshot["tasks"]["openarm-bi-reach"]["contract"]["output_signature"]["final_left_target_distance_mean"] >= 0.0
    assert snapshot["tasks"]["openarm-bi-reach"]["contract"]["output_signature"]["final_right_target_distance_mean"] >= 0.0
    assert snapshot["tasks"]["ur10-reach"]["contract"]["action_dim"] == 6
    assert snapshot["tasks"]["ur10-reach"]["contract"]["hotpath"] == expected_ur10e_hotpath
    assert snapshot["tasks"]["ur10-reach"]["contract"]["output_signature"]["final_target_distance_mean"] >= 0.0
    assert snapshot["tasks"]["ur10-reach"]["contract"]["output_signature"]["final_orientation_error_mean"] >= 0.0
    assert snapshot["tasks"]["ur10e-deploy-reach"]["contract"]["action_dim"] == 6
    assert snapshot["tasks"]["ur10e-deploy-reach"]["contract"]["hotpath"] == expected_ur10e_hotpath
    assert snapshot["tasks"]["ur10e-deploy-reach"]["contract"]["output_signature"]["final_target_distance_mean"] >= 0.0
    assert snapshot["tasks"]["ur10e-deploy-reach"]["contract"]["output_signature"]["final_orientation_error_mean"] >= 0.0
    assert snapshot["tasks"]["ur10e-gear-assembly-2f140"]["contract"]["action_dim"] == 6
    assert snapshot["tasks"]["ur10e-gear-assembly-2f140"]["contract"]["hotpath"] == expected_ur10e_hotpath
    assert snapshot["tasks"]["ur10e-gear-assembly-2f140"]["contract"]["output_signature"]["final_insert_depth_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["ur10e-gear-assembly-2f140"]["contract"]["output_signature"]["final_assembled_ratio"] <= 1.0
    assert snapshot["tasks"]["ur10e-gear-assembly-2f140"]["contract"]["output_signature"]["final_gear_type_mean"] >= 0.0
    assert snapshot["tasks"]["ur10e-gear-assembly-2f85"]["contract"]["action_dim"] == 6
    assert snapshot["tasks"]["ur10e-gear-assembly-2f85"]["contract"]["hotpath"] == expected_ur10e_hotpath
    assert snapshot["tasks"]["ur10e-gear-assembly-2f85"]["contract"]["output_signature"]["final_insert_depth_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["ur10e-gear-assembly-2f85"]["contract"]["output_signature"]["final_assembled_ratio"] <= 1.0
    assert snapshot["tasks"]["ur10e-gear-assembly-2f85"]["contract"]["output_signature"]["final_gear_type_mean"] >= 0.0
    assert snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["action_dim"] == 7
    assert snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["hotpath"] == expected_ur10e_hotpath
    assert snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["semantic_contract"] == "reduced-analytic-suction-stack"
    assert snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["output_signature"]["final_support_cube_height_mean"] > 0.0
    assert snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["output_signature"]["final_middle_stack_distance_mean"] >= 0.0
    assert snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["output_signature"]["final_top_stack_distance_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["output_signature"]["final_middle_stacked_ratio"] <= 1.0
    assert 0.0 <= snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["output_signature"]["final_top_stacked_ratio"] <= 1.0
    assert 0.0 <= snapshot["tasks"]["ur10-long-suction-stack"]["contract"]["output_signature"]["final_active_is_top_ratio"] <= 1.0
    assert snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["action_dim"] == 7
    assert snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["hotpath"] == expected_ur10e_hotpath
    assert snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["semantic_contract"] == "reduced-analytic-suction-stack"
    assert snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["output_signature"]["final_support_cube_height_mean"] > 0.0
    assert snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["output_signature"]["final_middle_stack_distance_mean"] >= 0.0
    assert snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["output_signature"]["final_top_stack_distance_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["output_signature"]["final_middle_stacked_ratio"] <= 1.0
    assert 0.0 <= snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["output_signature"]["final_top_stacked_ratio"] <= 1.0
    assert 0.0 <= snapshot["tasks"]["ur10-short-suction-stack"]["contract"]["output_signature"]["final_active_is_top_ratio"] <= 1.0
    assert snapshot["tasks"]["franka-lift"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-lift"]["contract"]["hotpath"] == expected_franka_lift_hotpath
    assert snapshot["tasks"]["openarm-lift"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["openarm-lift"]["contract"]["hotpath"] == expected_franka_lift_hotpath
    assert snapshot["tasks"]["openarm-lift"]["contract"]["output_signature"]["final_grasp_ratio"] >= 0.0
    assert snapshot["tasks"]["franka-teddy-bear-lift"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-teddy-bear-lift"]["contract"]["hotpath"] == expected_franka_lift_hotpath
    assert snapshot["tasks"]["franka-teddy-bear-lift"]["contract"]["output_signature"]["final_lift_gap_mean"] >= 0.0
    assert snapshot["tasks"]["franka-stack-instance-randomize"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-stack-instance-randomize"]["contract"]["hotpath"] == expected_franka_stack_hotpath
    assert snapshot["tasks"]["franka-stack-instance-randomize"]["contract"]["output_signature"][
        "final_support_variant_mean"
    ] >= 0.0
    assert snapshot["tasks"]["franka-stack-instance-randomize"]["contract"]["output_signature"][
        "final_movable_variant_mean"
    ] >= 0.0
    assert snapshot["tasks"]["franka-stack"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-stack"]["contract"]["hotpath"] == expected_franka_stack_hotpath
    assert snapshot["tasks"]["franka-stack"]["contract"]["output_signature"]["final_support_cube_height_mean"] > 0.0
    assert snapshot["tasks"]["franka-stack"]["contract"]["output_signature"]["final_stack_distance_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["franka-stack"]["contract"]["output_signature"]["final_stacked_ratio"] <= 1.0
    assert snapshot["tasks"]["franka-stack-rgb"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-stack-rgb"]["contract"]["hotpath"] == expected_franka_stack_rgb_hotpath
    assert snapshot["tasks"]["franka-stack-rgb"]["contract"]["output_signature"]["final_support_cube_height_mean"] > 0.0
    assert snapshot["tasks"]["franka-stack-rgb"]["contract"]["output_signature"]["final_middle_stack_distance_mean"] >= 0.0
    assert snapshot["tasks"]["franka-stack-rgb"]["contract"]["output_signature"]["final_top_stack_distance_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["franka-stack-rgb"]["contract"]["output_signature"]["final_middle_stacked_ratio"] <= 1.0
    assert 0.0 <= snapshot["tasks"]["franka-stack-rgb"]["contract"]["output_signature"]["final_top_stacked_ratio"] <= 1.0
    assert snapshot["tasks"]["franka-bin-stack"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-bin-stack"]["contract"]["hotpath"] == expected_franka_stack_rgb_hotpath
    assert snapshot["tasks"]["franka-bin-stack"]["contract"]["output_signature"]["final_bin_anchor_height_mean"] > 0.0
    assert snapshot["tasks"]["franka-bin-stack"]["contract"]["output_signature"]["final_support_bin_offset_mean"] >= 0.0
    assert snapshot["tasks"]["franka-cabinet"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-cabinet"]["contract"]["hotpath"] == get_franka_cabinet_hotpath_backend()
    assert snapshot["tasks"]["franka-cabinet"]["contract"]["output_signature"]["final_drawer_open_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["franka-cabinet"]["contract"]["output_signature"]["final_drawer_opened_ratio"] <= 1.0
    assert snapshot["tasks"]["franka-open-drawer"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["franka-open-drawer"]["contract"]["hotpath"] == get_franka_cabinet_hotpath_backend()
    assert snapshot["tasks"]["franka-open-drawer"]["contract"]["output_signature"]["final_drawer_open_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["franka-open-drawer"]["contract"]["output_signature"]["final_drawer_opened_ratio"] <= 1.0
    assert snapshot["tasks"]["openarm-open-drawer"]["contract"]["action_dim"] == 8
    assert snapshot["tasks"]["openarm-open-drawer"]["contract"]["hotpath"] == get_franka_cabinet_hotpath_backend()
    assert snapshot["tasks"]["openarm-open-drawer"]["contract"]["output_signature"]["final_drawer_open_mean"] >= 0.0
    assert 0.0 <= snapshot["tasks"]["openarm-open-drawer"]["contract"]["output_signature"]["final_drawer_opened_ratio"] <= 1.0
    assert snapshot["tasks"]["h1-rough"]["contract"]["sensor_scan_dim"] == 9
    assert snapshot["tasks"]["h1-flat-height-scan"]["contract"]["hotpath"] == expected_locomotion_hotpath
    assert snapshot["tasks"]["quadcopter"]["contract"]["output_signature"]["final_distance_to_goal_mean"] > 0.0


def test_semantic_drift_compare_matches_committed_baseline(tmp_path: Path):
    """The committed semantic baseline should still match the current deterministic benchmark contract."""
    benchmark_module = _load_module("benchmark_mac_tasks", "scripts/benchmarks/mlx/benchmark_mac_tasks.py")
    baseline_path = Path(__file__).resolve().parents[4] / "scripts" / "benchmarks" / "mlx" / "baselines" / "semantic-baseline.json"

    results = benchmark_module.run_benchmarks(
        benchmark_module.resolve_requested_tasks(None, "full"),
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=42,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )
    snapshot = build_semantic_drift_snapshot(results)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report = compare_semantic_drift(snapshot, baseline)

    assert report["passed"] is True
    assert report["failures"] == []


def test_semantic_drift_cli_emits_snapshot_and_report(tmp_path: Path, monkeypatch):
    """The drift CLI should write the current snapshot and a successful compare report."""
    benchmark_module = _load_module("benchmark_mac_tasks", "scripts/benchmarks/mlx/benchmark_mac_tasks.py")
    drift_module = _load_module("check_semantic_drift", "scripts/benchmarks/mlx/check_semantic_drift.py")
    results_path = tmp_path / "full-results.json"
    baseline_path = Path(__file__).resolve().parents[4] / "scripts" / "benchmarks" / "mlx" / "baselines" / "semantic-baseline.json"
    snapshot_path = tmp_path / "semantic-snapshot.json"
    report_path = tmp_path / "semantic-report.json"

    results = benchmark_module.run_benchmarks(
        benchmark_module.resolve_requested_tasks(None, "full"),
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=42,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_semantic_drift.py",
            "--results",
            str(results_path),
            "--baseline",
            str(baseline_path),
            "--snapshot-out",
            str(snapshot_path),
            "--report-out",
            str(report_path),
        ],
    )

    assert drift_module.main() == 0
    assert snapshot_path.exists()
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
