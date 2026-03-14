# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin CLI wrapper for evaluating the rough ANYmal-C MLX/mac-sim slice."""

from _task_support import run_eval_cli


def main() -> int:
    return run_eval_cli("anymal-c-rough")


if __name__ == "__main__":
    raise SystemExit(main())
