# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for creating prims in Omniverse."""

from __future__ import annotations

import importlib

_SEARCH_MODULES = (
    ".from_files",
    ".lights",
    ".materials",
    ".meshes",
    ".sensors",
    ".shapes",
    ".spawner_cfg",
    ".wrappers",
)


def __getattr__(name: str):
    for module_name in _SEARCH_MODULES:
        module = importlib.import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
