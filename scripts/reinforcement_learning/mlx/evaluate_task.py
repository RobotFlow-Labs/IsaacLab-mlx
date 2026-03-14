# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared evaluation and replay entrypoint for the MLX/mac-sim task slices."""

from __future__ import annotations

from _task_support import run_eval_cli


def main() -> int:
    return run_eval_cli()


if __name__ == "__main__":
    raise SystemExit(main())
