# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``direct``: These include single-file implementations of tasks.
- ``manager_based``: These include task implementations that use the manager-based API.
- ``utils``: These include utility functions for the tasks.

"""

import os

import gymnasium as gym
import toml

from isaaclab.backends import current_runtime

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from .utils.importer import import_packages

from .registry import ISAACSIM_ONLY_TASK_SPECS, LAZY_IMPORT_BLACKLIST, MAC_SAFE_TASK_SPECS


def _register_gym_specs(specs: tuple[dict[str, object], ...]) -> None:
    """Register the provided task specs if they are not already present."""
    for spec in specs:
        if spec["id"] in gym.registry:
            continue
        gym.register(
            id=spec["id"],
            entry_point=spec["entry_point"],
            disable_env_checker=True,
            kwargs=spec["kwargs"],
        )


def register_tasks() -> None:
    """Register task packages appropriate for the active runtime."""
    _register_gym_specs(ISAACSIM_ONLY_TASK_SPECS)

    if current_runtime().sim_backend == "isaacsim":
        import_packages(__name__, LAZY_IMPORT_BLACKLIST)
        return

    _register_gym_specs(MAC_SAFE_TASK_SPECS)


register_tasks()
