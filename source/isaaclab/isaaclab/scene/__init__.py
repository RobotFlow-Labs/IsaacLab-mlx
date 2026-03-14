# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for interactive scene definitions, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_MODULE_EXPORTS = {
    "InteractiveScene": (".interactive_scene", "InteractiveScene"),
    "InteractiveSceneCfg": (".interactive_scene_cfg", "InteractiveSceneCfg"),
}

__all__ = [*_MODULE_EXPORTS.keys()]


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.scene.{name}` currently requires `sim-backend=isaacsim`."
            " Scene interfaces for `mac-sim` are exposed progressively via backend capabilities."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
