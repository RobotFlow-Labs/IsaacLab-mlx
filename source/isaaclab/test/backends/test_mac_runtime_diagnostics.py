# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from isaaclab.backends import build_runtime_diagnostics_payload, resolve_runtime_selection
from isaaclab.backends.test_utils import require_mlx_runtime


def test_build_runtime_diagnostics_payload_reports_supported_surface():
    """Runtime diagnostics should expose the supported MLX/mac task surface and backend seams."""

    payload = build_runtime_diagnostics_payload(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert payload["runtime"]["compute_backend"] == "mlx"
    assert payload["sim"]["implementation"] == "generic-articulation-layer+task-specialized-analytic-slices"
    assert payload["sim"]["supported_tasks"]["current_mac_native_count"] >= 13
    assert payload["sim"]["supported_tasks"]["public_benchmark_groups"]["sensor-mac-native"] == [
        "cartpole-rgb-camera",
        "cartpole-depth-camera",
    ]
    assert payload["sim"]["supported_tasks"]["benchmark_task_groups"]["sensor-mac-native"] == [
        "cartpole-rgb-camera",
        "cartpole-depth-camera",
        "anymal-c-flat-height-scan",
        "h1-flat-height-scan",
    ]
    assert "training-mac-native" not in payload["sim"]["supported_tasks"]["public_benchmark_groups"]
    assert payload["sim"]["supported_tasks"]["benchmark_task_groups"]["training-mac-native"] == ["train-cartpole"]
    assert payload["sensor"]["capabilities"]["analytic_camera_tasks"] is True
    assert payload["sensor"]["generic_sensor_api"] == {
        "proprioception": True,
        "raycast": True,
        "cameras": False,
        "depth": False,
        "segmentation": False,
        "rgb": False,
    }
    assert payload["sensor"]["tooling_surface"] == {
        "analytic_camera_tasks": True,
        "external_stereo_capture": True,
        "synthetic_camera_tasks": True,
    }
    assert payload["sensor"]["tooling_sources"] == {
        "analytic_camera_tasks": "mac-native analytic task slices",
        "external_stereo_capture": "zed-sdk-mlx terminal-hosted capture path",
        "synthetic_camera_tasks": "task-local synthetic camera slices",
    }
    assert payload["planner"]["backend"] == "mac-planners"


def test_mac_runtime_diagnostics_script_writes_json(tmp_path: Path):
    """The runtime diagnostics CLI should emit a machine-checkable JSON artifact."""

    script = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "mac_runtime_diagnostics.py"
    output_path = tmp_path / "runtime-diagnostics.json"

    result = subprocess.run(
        [sys.executable, str(script), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert int(result.stdout.strip()) >= 13
    assert payload["runtime"]["supported_tasks"]["trainable_task_count"] >= 10
    assert payload["kernel"]["backend"] == "metal"


def test_mac_runtime_diagnostics_module_writes_json(tmp_path: Path):
    """The installed runtime diagnostics module entrypoint should emit the same JSON contract."""

    output_path = tmp_path / "runtime-diagnostics-module.json"

    result = subprocess.run(
        [sys.executable, "-m", "isaaclab.backends.runtime_cli", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert int(result.stdout.strip()) >= 13
    assert payload["runtime"]["supported_tasks"]["public_task_count"] >= 15
    assert payload["sim"]["generic_scene_runtime"] is True


def test_anymal_env_diagnostics_expose_articulated_runtime_contract():
    """Env-level diagnostics should expose the articulated mac-sim contract on a real locomotion backend."""
    require_mlx_runtime()
    from isaaclab.backends.mac_sim import MacAnymalCFlatEnv, MacAnymalCFlatEnvCfg, mac_env_diagnostics
    from isaaclab.backends.mac_sim.hotpath import get_locomotion_hotpath_backend

    env = MacAnymalCFlatEnv(MacAnymalCFlatEnvCfg(num_envs=4, seed=5, episode_length_s=0.5))

    payload = mac_env_diagnostics(env)
    sim_backend = payload["sim_backend"]

    assert sim_backend["backend"] == "mac-sim"
    assert sim_backend["capabilities"]["batched_stepping"] is True
    assert sim_backend["capabilities"]["articulated_rigid_bodies"] is True
    assert sim_backend["contract"]["articulations"] == {
        "joint_state_io": True,
        "root_state_io": True,
        "effort_targets": True,
        "batched_views": True,
    }
    assert sim_backend["root_state_shape"] == [4, 3]
    assert sim_backend["joint_state_shape"] == [4, env.cfg.action_space]
    assert sim_backend["subsystems"]["terrain"] is True
    assert sim_backend["subsystems"]["contacts"] is True
    assert sim_backend["subsystems"]["hotpath"] == get_locomotion_hotpath_backend()
    assert payload["terrain"]["num_envs"] == 4
    assert payload["contacts"]["history_length"] >= 1
    assert payload["contacts"]["hotpath_backend"] == "mlx-compiled"
