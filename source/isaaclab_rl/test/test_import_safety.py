# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-safety tests for the RL extension on the MLX/mac path."""

from __future__ import annotations

import importlib

from isaaclab.backends import resolve_runtime_selection, set_runtime_selection


def test_isaaclab_rl_modules_import_on_mlx_mac_path():
    """The RL extension root and lightweight wrappers should import without Isaac Sim."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_rl")
    importlib.import_module("isaaclab_rl.skrl")
    importlib.import_module("isaaclab_rl.sb3")
