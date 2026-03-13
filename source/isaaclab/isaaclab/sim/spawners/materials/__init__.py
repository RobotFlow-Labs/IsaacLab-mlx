# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn USD-based and PhysX-based materials."""

from __future__ import annotations

import importlib

_SEARCH_MODULES = (
    ".physics_materials",
    ".physics_materials_cfg",
    ".visual_materials",
    ".visual_materials_cfg",
)


def __getattr__(name: str):
    for module_name in _SEARCH_MODULES:
        module = importlib.import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
