# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the shared MLX/mac-sim trainer for the Factory gear-mesh slice."""

from __future__ import annotations

from _task_support import run_train_cli


def main() -> int:
    return run_train_cli("factory-gear-mesh")


if __name__ == "__main__":
    raise SystemExit(main())
