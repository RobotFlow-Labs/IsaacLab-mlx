# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with utilities, data collectors and environment wrappers."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "import_modules": (".importer", "import_modules"),
    "import_packages": (".importer", "import_packages"),
    "get_checkpoint_path": (".parse_cfg", "get_checkpoint_path"),
    "load_cfg_from_registry": (".parse_cfg", "load_cfg_from_registry"),
    "parse_env_cfg": (".parse_cfg", "parse_env_cfg"),
}

__all__ = [*_EXPORTS.keys()]


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
