# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray-caster pattern surfaces with config-safe lazy exports."""

from __future__ import annotations

import importlib

_CFG_EXPORTS = {
    "BpearlPatternCfg": (".patterns_cfg", "BpearlPatternCfg"),
    "GridPatternCfg": (".patterns_cfg", "GridPatternCfg"),
    "LidarPatternCfg": (".patterns_cfg", "LidarPatternCfg"),
    "PatternBaseCfg": (".patterns_cfg", "PatternBaseCfg"),
    "PinholeCameraPatternCfg": (".patterns_cfg", "PinholeCameraPatternCfg"),
}
_RUNTIME_EXPORTS = {
    "bpearl_pattern": (".patterns", "bpearl_pattern"),
    "grid_pattern": (".patterns", "grid_pattern"),
    "lidar_pattern": (".patterns", "lidar_pattern"),
    "pinhole_camera_pattern": (".patterns", "pinhole_camera_pattern"),
}

__all__ = [*_CFG_EXPORTS.keys(), *_RUNTIME_EXPORTS.keys()]


def __getattr__(name: str):
    target = _CFG_EXPORTS.get(name)
    if target is None:
        target = _RUNTIME_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
