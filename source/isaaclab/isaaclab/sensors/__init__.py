# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing sensor classes, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_MODULE_EXPORTS = {
    "SensorBase": (".sensor_base", "SensorBase"),
    "SensorBaseCfg": (".sensor_base_cfg", "SensorBaseCfg"),
    "Camera": (".camera", "Camera"),
    "CameraCfg": (".camera", "CameraCfg"),
    "CameraData": (".camera", "CameraData"),
    "TiledCamera": (".camera", "TiledCamera"),
    "TiledCameraCfg": (".camera", "TiledCameraCfg"),
    "ContactSensor": (".contact_sensor", "ContactSensor"),
    "ContactSensorCfg": (".contact_sensor", "ContactSensorCfg"),
    "ContactSensorData": (".contact_sensor", "ContactSensorData"),
    "FrameTransformer": (".frame_transformer", "FrameTransformer"),
    "FrameTransformerCfg": (".frame_transformer", "FrameTransformerCfg"),
    "FrameTransformerData": (".frame_transformer", "FrameTransformerData"),
    "OffsetCfg": (".frame_transformer", "OffsetCfg"),
    "Imu": (".imu", "Imu"),
    "ImuCfg": (".imu", "ImuCfg"),
    "ImuData": (".imu", "ImuData"),
    "patterns": (".ray_caster", "patterns"),
    "RayCaster": (".ray_caster", "RayCaster"),
    "RayCasterCfg": (".ray_caster", "RayCasterCfg"),
    "RayCasterData": (".ray_caster", "RayCasterData"),
    "RayCasterCamera": (".ray_caster", "RayCasterCamera"),
    "RayCasterCameraCfg": (".ray_caster", "RayCasterCameraCfg"),
    "MultiMeshRayCaster": (".ray_caster", "MultiMeshRayCaster"),
    "MultiMeshRayCasterCfg": (".ray_caster", "MultiMeshRayCasterCfg"),
    "MultiMeshRayCasterData": (".ray_caster", "MultiMeshRayCasterData"),
    "MultiMeshRayCasterCamera": (".ray_caster", "MultiMeshRayCasterCamera"),
    "MultiMeshRayCasterCameraCfg": (".ray_caster", "MultiMeshRayCasterCameraCfg"),
    "MultiMeshRayCasterCameraData": (".ray_caster", "MultiMeshRayCasterCameraData"),
}

__all__ = [*_MODULE_EXPORTS.keys()]


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.sensors.{name}` currently requires `sim-backend=isaacsim`."
            " Sensor interfaces on `mac-sim` are exposed progressively via backend capabilities."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
