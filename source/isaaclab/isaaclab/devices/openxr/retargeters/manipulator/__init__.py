# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka manipulator retargeting module."""

from __future__ import annotations

import importlib

_MODULE_EXPORTS = {
    "GripperRetargeter": (".gripper_retargeter", "GripperRetargeter"),
    "GripperRetargeterCfg": (".gripper_retargeter", "GripperRetargeterCfg"),
    "Se3AbsRetargeter": (".se3_abs_retargeter", "Se3AbsRetargeter"),
    "Se3AbsRetargeterCfg": (".se3_abs_retargeter", "Se3AbsRetargeterCfg"),
    "Se3RelRetargeter": (".se3_rel_retargeter", "Se3RelRetargeter"),
    "Se3RelRetargeterCfg": (".se3_rel_retargeter", "Se3RelRetargeterCfg"),
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
