# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR device and configuration surfaces, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "XrAnchorRotationMode": (".xr_cfg", "XrAnchorRotationMode"),
    "XrCfg": (".xr_cfg", "XrCfg"),
    "remove_camera_configs": (".xr_cfg", "remove_camera_configs"),
}
_SAFE_MODULES = {
    "retargeters": ".retargeters",
}
_ISAACSIM_EXPORTS = {
    "ManusVive": (".manus_vive", "ManusVive"),
    "ManusViveCfg": (".manus_vive", "ManusViveCfg"),
    "OpenXRDevice": (".openxr_device", "OpenXRDevice"),
    "OpenXRDeviceCfg": (".openxr_device", "OpenXRDeviceCfg"),
}

__all__ = [*_SAFE_EXPORTS.keys(), *_SAFE_MODULES.keys(), *_ISAACSIM_EXPORTS.keys()]


def __getattr__(name: str):
    target = _SAFE_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    safe_module = _SAFE_MODULES.get(name)
    if safe_module is not None:
        module = importlib.import_module(safe_module, __name__)
        globals()[name] = module
        return module

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.devices.openxr.{name}` currently requires `sim-backend=isaacsim`."
            " OpenXR config helpers stay import-safe on the `mac-sim` bootstrap path, while device runtimes remain"
            " Isaac Sim only."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
