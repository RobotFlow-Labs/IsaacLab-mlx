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


def test_build_runtime_diagnostics_payload_reports_supported_surface():
    """Runtime diagnostics should expose the supported MLX/mac task surface and backend seams."""

    payload = build_runtime_diagnostics_payload(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert payload["runtime"]["compute_backend"] == "mlx"
    assert payload["sim"]["implementation"] == "task-specialized-analytic-slices"
    assert payload["sim"]["supported_tasks"]["current_mac_native_count"] >= 13
    assert payload["sensor"]["capabilities"]["analytic_camera_tasks"] is True
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
