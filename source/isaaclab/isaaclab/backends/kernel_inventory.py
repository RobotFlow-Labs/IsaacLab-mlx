# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maintained inventory of upstream Warp/CUDA kernel surfaces for the MLX/mac port."""

from __future__ import annotations

from dataclasses import dataclass

CURRENT_MAC_NATIVE_TASKS = ("cartpole", "cart-double-pendulum", "quadcopter", "anymal-c-flat", "h1-flat")
"""Current benchmarked mac-native task slices that must stay in the MLX smoke suite."""


@dataclass(frozen=True)
class KernelInventoryItem:
    """Structured description of a Warp/CUDA kernel family to port or replace."""

    key: str
    family: str
    status: str
    target_tasks: tuple[str, ...]
    upstream_modules: tuple[str, ...]
    symbols: tuple[str, ...]
    replacement_strategy: str
    notes: str


KERNEL_PORT_INVENTORY = (
    KernelInventoryItem(
        key="raycast-mesh-ops",
        family="Warp mesh raycast kernels",
        status="next-sensor",
        target_tasks=("raycast-locomotion", "terrain-height-sampling", "sensor-raycast"),
        upstream_modules=(
            "source/isaaclab/isaaclab/utils/warp/ops.py",
            "source/isaaclab/isaaclab/utils/warp/kernels.py",
            "source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py",
            "source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_camera.py",
            "source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster.py",
            "source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_camera.py",
            "source/isaaclab/isaaclab/terrains/utils.py",
            "source/isaaclab/isaaclab/terrains/terrain_generator.py",
        ),
        symbols=(
            "convert_to_warp_mesh",
            "raycast_mesh",
            "raycast_single_mesh",
            "raycast_dynamic_meshes",
            "raycast_mesh_kernel",
            "raycast_dynamic_meshes_kernel",
        ),
        replacement_strategy="Use MLX for mesh preprocessing and a Metal-backed batched mesh-query kernel for the hot path.",
        notes="This is the first real kernel family gating raycast-driven locomotion and terrain sampling parity.",
    ),
    KernelInventoryItem(
        key="wrench-composer",
        family="Warp wrench composition kernels",
        status="next-manipulation",
        target_tasks=("manipulation-reach", "manipulation-lift", "external-wrench-controls"),
        upstream_modules=(
            "source/isaaclab/isaaclab/utils/warp/kernels.py",
            "source/isaaclab/isaaclab/utils/wrench_composer.py",
        ),
        symbols=("add_forces_and_torques_at_position", "set_forces_and_torques_at_position"),
        replacement_strategy="Replace with pure MLX vector math first; only drop to Metal if manipulation benchmarks make it hot.",
        notes="This is the first non-raycast kernel family likely to matter once manipulation tasks start landing on mac-sim.",
    ),
    KernelInventoryItem(
        key="fabric-transform-kernels",
        family="Warp Fabric transform kernels",
        status="inventory-only",
        target_tasks=("fabric-scene-views", "future-isaacsim-parity"),
        upstream_modules=(
            "source/isaaclab/isaaclab/utils/warp/fabric.py",
            "source/isaaclab/isaaclab/sim/views/xform_prim_view.py",
        ),
        symbols=(
            "set_view_to_fabric_array",
            "arange_k",
            "decompose_fabric_transformation_matrix_to_warp_arrays",
            "compose_fabric_transformation_matrix_from_warp_arrays",
        ),
        replacement_strategy="Keep CPU/bootstrap fallback first; only port to Metal if a Fabric-backed mac scene path becomes real.",
        notes="Important for engine-parity prep, but not on the immediate mac-native task critical path.",
    ),
    KernelInventoryItem(
        key="tiled-camera-reshape",
        family="Warp tiled-camera reshape kernel",
        status="later-camera",
        target_tasks=("rgb-camera", "depth-camera", "segmentation-camera"),
        upstream_modules=(
            "source/isaaclab/isaaclab/utils/warp/kernels.py",
            "source/isaaclab/isaaclab/sensors/camera/tiled_camera.py",
        ),
        symbols=("reshape_tiled_image",),
        replacement_strategy="Start with MLX reshape/transposes; add a Metal kernel only if camera ingestion becomes a measured hotspot.",
        notes="This is the key reshape path behind tiled camera parity, but it is deferred until sensor work starts.",
    ),
)
