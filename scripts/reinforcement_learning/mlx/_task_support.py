# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin bridge to the installed MLX/mac CLI module."""

from __future__ import annotations

from isaaclab_rl.mlx_cli import TASK_PREFIXES, _write_json, run_eval_cli, run_train_cli

__all__ = ["TASK_PREFIXES", "_write_json", "run_eval_cli", "run_train_cli"]
