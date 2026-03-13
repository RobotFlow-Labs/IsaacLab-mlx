# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing simulation-specific functionalities."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "PhysxCfg": (".simulation_cfg", "PhysxCfg"),
    "RenderCfg": (".simulation_cfg", "RenderCfg"),
    "SimulationCfg": (".simulation_cfg", "SimulationCfg"),
}
_ISAACSIM_EXPORTS = {
    "SimulationContext": (".simulation_context", "SimulationContext"),
    "build_simulation_context": (".simulation_context", "build_simulation_context"),
}
_SEARCH_MODULES = (
    ".converters",
    ".schemas",
    ".spawners",
    ".utils",
    ".views",
)

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
            f"`isaaclab.sim.{name}` currently requires `sim-backend=isaacsim`."
            " Only simulation configuration objects are available in the `mac-sim` bootstrap path today."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    for module_name in _SEARCH_MODULES:
        module = importlib.import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
