# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for Warp-based ray-cast sensor.

The sub-module contains two implementations of the ray-cast sensor:

- :class:`isaaclab.sensors.ray_caster.RayCaster`: A basic ray-cast sensor that can be used to ray-cast against a single mesh.
- :class:`isaaclab.sensors.ray_caster.MultiMeshRayCaster`: A multi-mesh ray-cast sensor that can be used to ray-cast against
  multiple meshes. For these meshes, it tracks their transformations and updates the warp meshes accordingly.

Corresponding camera implementations are also provided for each of the sensor implementations. Internally, they perform
the same ray-casting operations as the sensor implementations, but return the results as images.
"""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "MultiMeshRayCasterCameraCfg": (".multi_mesh_ray_caster_camera_cfg", "MultiMeshRayCasterCameraCfg"),
    "MultiMeshRayCasterCfg": (".multi_mesh_ray_caster_cfg", "MultiMeshRayCasterCfg"),
    "RayCasterCameraCfg": (".ray_caster_camera_cfg", "RayCasterCameraCfg"),
    "RayCasterCfg": (".ray_caster_cfg", "RayCasterCfg"),
}
_SAFE_MODULES = {
    "patterns": ".patterns",
}
_ISAACSIM_EXPORTS = {
    "MultiMeshRayCaster": (".multi_mesh_ray_caster", "MultiMeshRayCaster"),
    "MultiMeshRayCasterCamera": (".multi_mesh_ray_caster_camera", "MultiMeshRayCasterCamera"),
    "MultiMeshRayCasterCameraData": (".multi_mesh_ray_caster_camera_data", "MultiMeshRayCasterCameraData"),
    "MultiMeshRayCasterData": (".multi_mesh_ray_caster_data", "MultiMeshRayCasterData"),
    "RayCaster": (".ray_caster", "RayCaster"),
    "RayCasterCamera": (".ray_caster_camera", "RayCasterCamera"),
    "RayCasterData": (".ray_caster_data", "RayCasterData"),
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
            f"`isaaclab.sensors.ray_caster.{name}` currently requires `sim-backend=isaacsim`."
            " Ray-caster configs stay import-safe on `mac-sim`, while runtime ray-cast sensors remain Isaac Sim only."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
