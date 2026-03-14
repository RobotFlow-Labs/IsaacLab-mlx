# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke the mac planner compatibility seam without requiring Isaac Sim or cuRobo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.backends import (
    JointMotionPlanRequest,
    PlannerWorldObstacle,
    PlannerWorldState,
    create_planner_backend,
    resolve_runtime_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test for the mac planner compatibility backend.")
    parser.add_argument("output", type=Path, help="Path to the output JSON artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    planner = create_planner_backend(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    planner.update_world_state(
        PlannerWorldState(
            obstacles=(PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),),
        )
    )
    plan = planner.plan_joint_motion(
        JointMotionPlanRequest(
            joint_names=("joint_1", "joint_2", "joint_3"),
            start_positions=(0.0, -0.5, 0.25),
            goal_positions=(0.8, 0.25, -0.1),
            num_waypoints=6,
            duration_s=1.5,
        )
    )

    payload = {
        "planner": planner.state_dict(),
        "plan": plan.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["plan"]["waypoint_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
