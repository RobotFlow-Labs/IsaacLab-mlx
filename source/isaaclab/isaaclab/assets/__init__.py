# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for simulation assets, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_MODULE_EXPORTS = {
    "Articulation": (".articulation", "Articulation"),
    "ArticulationCfg": (".articulation", "ArticulationCfg"),
    "ArticulationData": (".articulation", "ArticulationData"),
    "AssetBase": (".asset_base", "AssetBase"),
    "AssetBaseCfg": (".asset_base_cfg", "AssetBaseCfg"),
    "DeformableObject": (".deformable_object", "DeformableObject"),
    "DeformableObjectCfg": (".deformable_object", "DeformableObjectCfg"),
    "DeformableObjectData": (".deformable_object", "DeformableObjectData"),
    "RigidObject": (".rigid_object", "RigidObject"),
    "RigidObjectCfg": (".rigid_object", "RigidObjectCfg"),
    "RigidObjectData": (".rigid_object", "RigidObjectData"),
    "RigidObjectCollection": (".rigid_object_collection", "RigidObjectCollection"),
    "RigidObjectCollectionCfg": (".rigid_object_collection", "RigidObjectCollectionCfg"),
    "RigidObjectCollectionData": (".rigid_object_collection", "RigidObjectCollectionData"),
    "SurfaceGripper": (".surface_gripper", "SurfaceGripper"),
    "SurfaceGripperCfg": (".surface_gripper", "SurfaceGripperCfg"),
}

__all__ = [*_MODULE_EXPORTS.keys()]


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.assets.{name}` currently requires `sim-backend=isaacsim`."
            " The `mac-sim` path will expose asset interfaces as capabilities are implemented."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
