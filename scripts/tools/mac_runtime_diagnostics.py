# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Emit a machine-checkable runtime diagnostics snapshot for the MLX/mac path."""

from __future__ import annotations

from isaaclab.backends.runtime_cli import runtime_diagnostics_main as main


if __name__ == "__main__":
    raise SystemExit(main())
