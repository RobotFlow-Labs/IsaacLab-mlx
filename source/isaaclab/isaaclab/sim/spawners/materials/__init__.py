# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn USD-based and PhysX-based materials."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "PhysicsMaterialCfg": (".physics_materials_cfg", "PhysicsMaterialCfg"),
    "RigidBodyMaterialCfg": (".physics_materials_cfg", "RigidBodyMaterialCfg"),
    "DeformableBodyMaterialCfg": (".physics_materials_cfg", "DeformableBodyMaterialCfg"),
    "VisualMaterialCfg": (".visual_materials_cfg", "VisualMaterialCfg"),
    "PreviewSurfaceCfg": (".visual_materials_cfg", "PreviewSurfaceCfg"),
    "MdlFileCfg": (".visual_materials_cfg", "MdlFileCfg"),
    "GlassMdlCfg": (".visual_materials_cfg", "GlassMdlCfg"),
}
_ISAACSIM_EXPORTS = {
    "spawn_rigid_body_material": (".physics_materials", "spawn_rigid_body_material"),
    "spawn_deformable_body_material": (".physics_materials", "spawn_deformable_body_material"),
    "spawn_preview_surface": (".visual_materials", "spawn_preview_surface"),
    "spawn_from_mdl_file": (".visual_materials", "spawn_from_mdl_file"),
}

__all__ = [*_SAFE_EXPORTS.keys(), *_ISAACSIM_EXPORTS.keys()]


def __getattr__(name: str):
    target = _SAFE_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.sim.spawners.materials.{name}` currently requires `sim-backend=isaacsim`."
            " Material configuration objects are available in the `mac-sim` bootstrap path."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
