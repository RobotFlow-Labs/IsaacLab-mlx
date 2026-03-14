# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Submodule for different interpolation methods."""

from __future__ import annotations

import importlib

_MODULE_EXPORTS = {
    "LinearInterpolation": (".linear_interpolation", "LinearInterpolation"),
}

__all__ = list(_MODULE_EXPORTS.keys())


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
