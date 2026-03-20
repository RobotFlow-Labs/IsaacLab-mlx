# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the shared MLX/mac-sim evaluator for the Factory peg-insert slice."""

from __future__ import annotations

from _task_support import run_eval_cli


def main() -> int:
    return run_eval_cli("factory-peg-insert")


if __name__ == "__main__":
    raise SystemExit(main())
