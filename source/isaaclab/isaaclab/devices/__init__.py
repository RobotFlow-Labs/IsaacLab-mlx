# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for teleoperation device interfaces, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_MODULE_EXPORTS = {
    "DeviceBase": (".device_base", "DeviceBase"),
    "DeviceCfg": (".device_base", "DeviceCfg"),
    "DevicesCfg": (".device_base", "DevicesCfg"),
    "Se2Gamepad": (".gamepad", "Se2Gamepad"),
    "Se2GamepadCfg": (".gamepad", "Se2GamepadCfg"),
    "Se3Gamepad": (".gamepad", "Se3Gamepad"),
    "Se3GamepadCfg": (".gamepad", "Se3GamepadCfg"),
    "HaplyDevice": (".haply", "HaplyDevice"),
    "HaplyDeviceCfg": (".haply", "HaplyDeviceCfg"),
    "Se2Keyboard": (".keyboard", "Se2Keyboard"),
    "Se2KeyboardCfg": (".keyboard", "Se2KeyboardCfg"),
    "Se3Keyboard": (".keyboard", "Se3Keyboard"),
    "Se3KeyboardCfg": (".keyboard", "Se3KeyboardCfg"),
    "ManusVive": (".openxr", "ManusVive"),
    "ManusViveCfg": (".openxr", "ManusViveCfg"),
    "OpenXRDevice": (".openxr", "OpenXRDevice"),
    "OpenXRDeviceCfg": (".openxr", "OpenXRDeviceCfg"),
    "RetargeterBase": (".retargeter_base", "RetargeterBase"),
    "RetargeterCfg": (".retargeter_base", "RetargeterCfg"),
    "Se2SpaceMouse": (".spacemouse", "Se2SpaceMouse"),
    "Se2SpaceMouseCfg": (".spacemouse", "Se2SpaceMouseCfg"),
    "Se3SpaceMouse": (".spacemouse", "Se3SpaceMouse"),
    "Se3SpaceMouseCfg": (".spacemouse", "Se3SpaceMouseCfg"),
    "create_teleop_device": (".teleop_device_factory", "create_teleop_device"),
}

__all__ = [*_MODULE_EXPORTS.keys()]


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.devices.{name}` currently requires `sim-backend=isaacsim`."
            " Device interfaces for `mac-sim` are exposed progressively via backend capabilities."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
