# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin CLI wrapper for the MLX/mac-sim OpenArm open-drawer slice."""

from __future__ import annotations

try:
    from ._task_support import run_train_cli
except ImportError:  # pragma: no cover - direct script execution
    from _task_support import run_train_cli


def main() -> int:
    return run_train_cli("openarm-open-drawer")


if __name__ == "__main__":
    raise SystemExit(main())
