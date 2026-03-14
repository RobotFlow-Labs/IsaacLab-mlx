# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for marker utilities, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_CORE_EXPORTS = {
    "VisualizationMarkers": (".visualization_markers", "VisualizationMarkers"),
    "VisualizationMarkersCfg": (".visualization_markers", "VisualizationMarkersCfg"),
}
_SEARCH_MODULES = (".config",)

__all__ = [*_CORE_EXPORTS.keys()]


def __getattr__(name: str):
    target = _CORE_EXPORTS.get(name)
    if target is not None:
        if current_runtime().sim_backend != "isaacsim":
            raise UnsupportedBackendError(
                f"`isaaclab.markers.{name}` currently requires `sim-backend=isaacsim`."
                " Marker interfaces for `mac-sim` are exposed progressively via backend capabilities."
            )
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.markers.{name}` currently requires `sim-backend=isaacsim`."
            " Marker interfaces for `mac-sim` are exposed progressively via backend capabilities."
        )

    for module_name in _SEARCH_MODULES:
        module = importlib.import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
