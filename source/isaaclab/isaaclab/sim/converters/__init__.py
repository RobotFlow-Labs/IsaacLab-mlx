# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing converters for converting various file types to USD.

In order to support direct loading of various file types into Omniverse, we provide a set of
converters that can convert the file into a USD file. The converters are implemented as
sub-classes of the :class:`AssetConverterBase` class.

The following converters are currently supported:

* :class:`UrdfConverter`: Converts a URDF file into a USD file.
* :class:`MeshConverter`: Converts a mesh file into a USD file. This supports OBJ, STL and FBX files.

"""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "AssetConverterBaseCfg": (".asset_converter_base_cfg", "AssetConverterBaseCfg"),
    "MeshConverterCfg": (".mesh_converter_cfg", "MeshConverterCfg"),
    "MjcfConverterCfg": (".mjcf_converter_cfg", "MjcfConverterCfg"),
    "UrdfConverterCfg": (".urdf_converter_cfg", "UrdfConverterCfg"),
}
_ISAACSIM_EXPORTS = {
    "AssetConverterBase": (".asset_converter_base", "AssetConverterBase"),
    "MeshConverter": (".mesh_converter", "MeshConverter"),
    "MjcfConverter": (".mjcf_converter", "MjcfConverter"),
    "UrdfConverter": (".urdf_converter", "UrdfConverter"),
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
            f"`isaaclab.sim.converters.{name}` currently requires `sim-backend=isaacsim`."
            " Converter configuration objects remain import-safe in the `mac-sim` bootstrap path."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
