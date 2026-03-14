# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build or compare semantic drift snapshots for the MLX/mac benchmark suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.backends import build_semantic_drift_snapshot, compare_semantic_drift


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or compare MLX/mac semantic drift snapshots.")
    parser.add_argument("--results", type=Path, required=True, help="Benchmark results JSON emitted by benchmark_mac_tasks.py")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Committed semantic baseline JSON used for nightly drift comparisons.",
    )
    parser.add_argument("--snapshot-out", type=Path, default=None, help="Optional path for the current semantic snapshot.")
    parser.add_argument("--report-out", type=Path, default=None, help="Optional path for the compare report.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write the current semantic snapshot to --baseline instead of comparing against it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    snapshot = build_semantic_drift_snapshot(results)

    if args.snapshot_out is not None:
        _write_json(args.snapshot_out, snapshot)

    if args.write_baseline:
        _write_json(args.baseline, snapshot)
        print(json.dumps({"baseline": str(args.baseline), "task_count": snapshot["task_count"]}, sort_keys=True))
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = compare_semantic_drift(snapshot, baseline)
    if args.report_out is not None:
        _write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
