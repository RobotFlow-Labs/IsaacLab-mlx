# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin CLI wrapper for training the rough ANYmal-C MLX/mac-sim slice."""

from _task_support import run_train_cli


def main() -> int:
    return run_train_cli(default_task="anymal-c-rough")


if __name__ == "__main__":
    raise SystemExit(main())
