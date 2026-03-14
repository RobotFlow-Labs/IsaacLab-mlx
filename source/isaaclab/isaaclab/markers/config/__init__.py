# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg, CylinderCfg, SphereCfg
from isaaclab.utils.nucleus import ISAAC_NUCLEUS_DIR

##
# Sensors.
##

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": SphereCfg(
            radius=0.02,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
"""Configuration for the ray-caster marker."""


CONTACT_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact": SphereCfg(
            radius=0.02,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "no_contact": SphereCfg(
            radius=0.02,
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            visible=False,
        ),
    },
)
"""Configuration for the contact sensor marker."""

DEFORMABLE_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target": SphereCfg(
            radius=0.02,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.8)),
        ),
    },
)
"""Configuration for the deformable object's kinematic target marker."""

VISUO_TACTILE_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "tacsl_pts": SphereCfg(
            radius=0.0002,
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    },
)
"""Configuration for the visuo-tactile sensor marker."""

##
# Frames.
##

FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        ),
        "connecting_line": CylinderCfg(
            radius=0.002,
            height=1.0,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
        ),
    }
)
"""Configuration for the frame marker."""


RED_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
"""Configuration for the red arrow marker (along x-direction)."""


BLUE_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)
"""Configuration for the blue arrow marker (along x-direction)."""

GREEN_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)
"""Configuration for the green arrow marker (along x-direction)."""


##
# Goals.
##

CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
"""Configuration for the cuboid marker."""

SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": SphereCfg(
            radius=0.05,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
"""Configuration for the sphere marker."""

POSITION_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target_far": SphereCfg(
            radius=0.01,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_near": SphereCfg(
            radius=0.01,
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "target_invisible": SphereCfg(
            radius=0.01,
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            visible=False,
        ),
    }
)
"""Configuration for the end-effector tracking marker."""
