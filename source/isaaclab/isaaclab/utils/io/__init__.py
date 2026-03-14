# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Submodules for files IO operations.
"""

from .yaml import dump_yaml, load_yaml

__all__ = ["dump_yaml", "load_yaml", "load_torchscript_model"]


def __getattr__(name: str):
    if name != "load_torchscript_model":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .torchscript import load_torchscript_model

    globals()[name] = load_torchscript_model
    return load_torchscript_model
