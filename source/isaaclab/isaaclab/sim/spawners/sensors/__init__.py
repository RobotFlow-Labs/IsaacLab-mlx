# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn sensors in the simulation.

Currently, the following sensors are supported:

* Camera: A USD camera prim with settings for pinhole or fisheye projections.

"""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "FisheyeCameraCfg": (".sensors_cfg", "FisheyeCameraCfg"),
    "PinholeCameraCfg": (".sensors_cfg", "PinholeCameraCfg"),
}
_RUNTIME_EXPORTS = {
    "spawn_camera": (".sensors", "spawn_camera"),
}

__all__ = [*_SAFE_EXPORTS.keys(), *_RUNTIME_EXPORTS.keys()]


def __getattr__(name: str):
    target = _SAFE_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    target = _RUNTIME_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.sim.spawners.sensors.{name}` currently requires `sim-backend=isaacsim`."
            " Camera spawner configs stay import-safe on `mac-sim`, while runtime sensor spawning remains Isaac Sim only."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
