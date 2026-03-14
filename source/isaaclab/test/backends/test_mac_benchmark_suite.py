# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark-suite contract tests for the mac-native MLX task slices."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[4] / "scripts" / "benchmarks" / "mlx" / "benchmark_mac_tasks.py"
    spec = importlib.util.spec_from_file_location("benchmark_mac_tasks", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_mac_native_benchmark_group_is_stable():
    """The benchmark suite should keep all current mac-native slices in one named task group."""
    benchmark_module = _load_benchmark_module()

    assert benchmark_module.CURRENT_MAC_NATIVE_TASKS == (
        "cartpole",
        "cart-double-pendulum",
        "quadcopter",
        "anymal-c-flat",
        "h1-flat",
    )
    assert benchmark_module.resolve_requested_tasks(None, "current-mac-native") == benchmark_module.CURRENT_MAC_NATIVE_TASKS


def test_run_benchmarks_covers_all_current_mac_native_tasks(tmp_path: Path):
    """The benchmark harness should emit diagnostics for every current mac-native task slice."""
    benchmark_module = _load_benchmark_module()

    results = benchmark_module.run_benchmarks(
        benchmark_module.CURRENT_MAC_NATIVE_TASKS,
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=7,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )

    assert results["task_group"] == "current-mac-native"
    assert results["tasks"] == list(benchmark_module.CURRENT_MAC_NATIVE_TASKS)
    assert results["cpu_fallback_detected"] is False
    assert results["cpu_fallback_tasks"] == []
    assert [benchmark["task"] for benchmark in results["benchmarks"]] == list(benchmark_module.CURRENT_MAC_NATIVE_TASKS)

    for benchmark in results["benchmarks"]:
        assert benchmark["runtime"]["compute_backend"] == "mlx"
        assert benchmark["runtime"]["sim_backend"] == "mac-sim"
        assert benchmark["cpu_fallback"]["detected"] is False
        assert benchmark["cpu_fallback"]["active_kernel_backend"] == "metal"
        assert "diagnostics" in benchmark
        assert benchmark["diagnostics"]["sim_backend"]["backend"] == "mac-sim"
        if benchmark["task"] in {"anymal-c-flat", "h1-flat"}:
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == "mlx-compiled"
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_root_height_mean",
                "final_root_lin_vel_norm_mean",
                "final_root_ang_vel_norm_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_joint_acc_abs_mean",
                "final_applied_torque_abs_mean",
                "final_contact_count",
            }
            assert all(math.isfinite(value) for value in benchmark["output_signature"].values())
