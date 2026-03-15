# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Emit a machine-checkable runtime diagnostics snapshot for the MLX/mac path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.backends import build_runtime_diagnostics_payload, resolve_runtime_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write runtime diagnostics for the active IsaacLab backend selection.")
    parser.add_argument("output", type=Path, help="Path to the JSON diagnostics artifact.")
    parser.add_argument("--compute-backend", default="mlx")
    parser.add_argument("--sim-backend", default="mac-sim")
    parser.add_argument("--kernel-backend", default="metal")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = resolve_runtime_selection(
        compute_backend=args.compute_backend,
        sim_backend=args.sim_backend,
        kernel_backend=args.kernel_backend,
        device=args.device,
    )
    payload = build_runtime_diagnostics_payload(runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["runtime"]["supported_tasks"]["public_task_count"] if "supported_tasks" in payload["runtime"] else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
