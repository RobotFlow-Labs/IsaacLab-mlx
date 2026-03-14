# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with implementation of manager terms.

The functions can be provided to different managers that are responsible for the
different aspects of the MDP. These include the observation, reward, termination,
actions, events and curriculum managers.

The terms are defined under the ``envs`` module because they are used to define
the environment. However, they are not part of the environment directly, but
are used to define the environment through their managers.

"""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_MODULES = (".actions",)
_ISAACSIM_MODULES = (
    ".commands",
    ".curriculums",
    ".events",
    ".observations",
    ".recorders",
    ".rewards",
    ".terminations",
)
_OPTIONAL_IMPORT_PREFIXES = ("pink", "omni", "isaacsim", "warp", "torch", "carb", "pxr")

__all__ = ["actions", "commands", "curriculums", "events", "observations", "recorders", "rewards", "terminations"]


def _import_optional(module_name: str):
    try:
        return importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith(_OPTIONAL_IMPORT_PREFIXES):
            return None
        raise


def __getattr__(name: str):
    if name in __all__:
        if name != "actions" and current_runtime().sim_backend != "isaacsim":
            raise UnsupportedBackendError(
                f"`isaaclab.envs.mdp.{name}` currently requires `sim-backend=isaacsim`."
                " The `mac-sim` bootstrap path currently exposes action configuration surfaces first."
            )
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    for module_name in _SAFE_MODULES:
        module = _import_optional(module_name)
        if module is not None and hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.envs.mdp.{name}` currently requires `sim-backend=isaacsim`."
            " The `mac-sim` bootstrap path currently exposes action configuration surfaces first."
        )

    for module_name in _ISAACSIM_MODULES:
        module = _import_optional(module_name)
        if module is not None and hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
