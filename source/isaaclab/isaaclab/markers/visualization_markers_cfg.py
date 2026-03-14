# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class VisualizationMarkersCfg:
    """A class to configure a :class:`isaaclab.markers.VisualizationMarkers`."""

    prim_path: str = MISSING
    """The prim path where the :class:`UsdGeom.PointInstancer` will be created."""

    markers: dict[str, SpawnerCfg] = MISSING
    """The dictionary of marker configurations."""
