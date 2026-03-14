# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera surfaces, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "CameraCfg": (".camera_cfg", "CameraCfg"),
    "TiledCameraCfg": (".tiled_camera_cfg", "TiledCameraCfg"),
}
_ISAACSIM_EXPORTS = {
    "Camera": (".camera", "Camera"),
    "CameraData": (".camera_data", "CameraData"),
    "TiledCamera": (".tiled_camera", "TiledCamera"),
}
_ISAACSIM_MODULES = {
    "utils": ".utils",
}

__all__ = [*_SAFE_EXPORTS.keys(), *_ISAACSIM_EXPORTS.keys(), *_ISAACSIM_MODULES.keys()]


def __getattr__(name: str):
    target = _SAFE_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.sensors.camera.{name}` currently requires `sim-backend=isaacsim`."
            " Camera configs stay import-safe on `mac-sim`, while runtime camera implementations remain Isaac Sim only."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    module_name = _ISAACSIM_MODULES.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __name__)
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
