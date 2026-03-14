# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing utilities for common operations and helper functions."""

from __future__ import annotations

import importlib

from .configclass import configclass
from .timer import Timer

_SEARCH_MODULES = (
    ".dict",
    ".string",
    ".version",
    ".array",
    ".buffers",
    ".interpolation",
    ".logger",
    ".mesh",
    ".modifiers",
    ".types",
)

__all__ = ["configclass", "Timer"]
_OPTIONAL_IMPORT_PREFIXES = ("omni", "isaacsim", "warp", "torch", "carb", "pxr")


def __getattr__(name: str):
    for module_name in _SEARCH_MODULES:
        try:
            module = importlib.import_module(module_name, __name__)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith(_OPTIONAL_IMPORT_PREFIXES):
                continue
            raise
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
