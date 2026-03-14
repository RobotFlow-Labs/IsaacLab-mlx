# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning primitive shapes in the simulation."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "CapsuleCfg": (".shapes_cfg", "CapsuleCfg"),
    "ConeCfg": (".shapes_cfg", "ConeCfg"),
    "CuboidCfg": (".shapes_cfg", "CuboidCfg"),
    "CylinderCfg": (".shapes_cfg", "CylinderCfg"),
    "ShapeCfg": (".shapes_cfg", "ShapeCfg"),
    "SphereCfg": (".shapes_cfg", "SphereCfg"),
}
_ISAACSIM_EXPORTS = {
    "spawn_capsule": (".shapes", "spawn_capsule"),
    "spawn_cone": (".shapes", "spawn_cone"),
    "spawn_cuboid": (".shapes", "spawn_cuboid"),
    "spawn_cylinder": (".shapes", "spawn_cylinder"),
    "spawn_sphere": (".shapes", "spawn_sphere"),
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
            f"`isaaclab.sim.spawners.shapes.{name}` currently requires `sim-backend=isaacsim`."
            " Shape configuration objects remain import-safe in the `mac-sim` bootstrap path."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
