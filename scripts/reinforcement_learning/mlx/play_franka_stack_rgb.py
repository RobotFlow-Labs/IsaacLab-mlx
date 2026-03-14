# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin CLI wrapper for replaying the MLX three-cube Franka stack slice."""

from __future__ import annotations

from _task_support import run_eval_cli


def main() -> int:
    return run_eval_cli(default_task="franka-stack-rgb")


if __name__ == "__main__":
    raise SystemExit(main())
