# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various action terms that can be used in the environment."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_MODULES = (
    ".actions_cfg",
    ".rmpflow_actions_cfg",
    ".pink_actions_cfg",
)
_ISAACSIM_MODULES = (
    ".binary_joint_actions",
    ".joint_actions",
    ".joint_actions_to_limits",
    ".non_holonomic_actions",
    ".surface_gripper_actions",
    ".task_space_actions",
    ".rmpflow_task_space_actions",
    ".pink_task_space_actions",
)
_OPTIONAL_IMPORT_PREFIXES = ("pink", "omni", "isaacsim", "warp", "torch", "carb", "pxr")


def _import_optional(module_name: str):
    try:
        return importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith(_OPTIONAL_IMPORT_PREFIXES):
            return None
        raise


def __getattr__(name: str):
    for module_name in _SAFE_MODULES:
        module = _import_optional(module_name)
        if module is not None and hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.envs.mdp.actions.{name}` currently requires `sim-backend=isaacsim`."
            " Action configuration objects remain import-safe in the `mac-sim` bootstrap path."
        )

    for module_name in _ISAACSIM_MODULES:
        module = _import_optional(module_name)
        if module is not None and hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
