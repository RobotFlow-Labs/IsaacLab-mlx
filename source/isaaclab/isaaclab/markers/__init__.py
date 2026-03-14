# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for marker utilities, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "VisualizationMarkersCfg": (".visualization_markers_cfg", "VisualizationMarkersCfg"),
}
_SAFE_MODULES = {
    "config": ".config",
}
_ISAACSIM_EXPORTS = {
    "VisualizationMarkers": (".visualization_markers", "VisualizationMarkers"),
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
            f"`isaaclab.markers.{name}` currently requires `sim-backend=isaacsim`."
            " Marker interfaces for `mac-sim` are exposed progressively via backend capabilities."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
