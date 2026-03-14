# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn assets from files.

Currently, the following spawners are supported:

* :class:`UsdFileCfg`: Spawn an asset from a USD file.
* :class:`UrdfFileCfg`: Spawn an asset from a URDF file.
* :class:`GroundPlaneCfg`: Spawn a ground plane using the grid-world USD file.

"""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "GroundPlaneCfg": (".from_files_cfg", "GroundPlaneCfg"),
    "MjcfFileCfg": (".from_files_cfg", "MjcfFileCfg"),
    "UrdfFileCfg": (".from_files_cfg", "UrdfFileCfg"),
    "UsdFileCfg": (".from_files_cfg", "UsdFileCfg"),
    "UsdFileWithCompliantContactCfg": (".from_files_cfg", "UsdFileWithCompliantContactCfg"),
}
_ISAACSIM_EXPORTS = {
    "spawn_from_mjcf": (".from_files", "spawn_from_mjcf"),
    "spawn_from_urdf": (".from_files", "spawn_from_urdf"),
    "spawn_from_usd": (".from_files", "spawn_from_usd"),
    "spawn_from_usd_with_compliant_contact_material": (
        ".from_files",
        "spawn_from_usd_with_compliant_contact_material",
    ),
    "spawn_ground_plane": (".from_files", "spawn_ground_plane"),
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
            f"`isaaclab.sim.spawners.from_files.{name}` currently requires `sim-backend=isaacsim`."
            " File-based spawner configuration objects remain import-safe in the `mac-sim` bootstrap path."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
