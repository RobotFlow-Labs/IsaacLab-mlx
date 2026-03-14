# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the mac-native analytic raycast substrate."""

from __future__ import annotations

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import MacPlaneRaycastSensor, MacPlaneTerrain  # noqa: E402


def test_plane_raycast_hits_ground_with_expected_distance_and_normal():
    """Downward rays should hit the analytic plane at the expected distance."""
    terrain = MacPlaneTerrain(2, env_spacing=4.0, origin_height=0.1)
    sensor = MacPlaneRaycastSensor(terrain, offsets_xy=((0.0, 0.0),), max_distance=3.0)
    origins = mx.array([[0.0, 0.0, 1.1], [0.0, 0.0, 1.6]], dtype=mx.float32)

    result = sensor.raycast(origins)

    assert bool(mx.all(result["hit_mask"]).item())
    assert bool(mx.allclose(result["hit_distances"], mx.array([1.0, 1.5], dtype=mx.float32)).item())
    assert bool(
        mx.allclose(
            result["hit_normals_w"],
            mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32),
        ).item()
    )


def test_height_scan_matches_plane_height_samples():
    """Height-scan distances should match the plane clearance for each origin."""
    terrain = MacPlaneTerrain(2, env_spacing=4.0, tile_size=(8.0, 8.0), origin_height=0.0)
    sensor = MacPlaneRaycastSensor(terrain, offsets_xy=((-0.25, 0.0), (0.0, 0.0), (0.25, 0.0)), max_distance=2.0)
    roots = mx.array([[0.0, 0.0, 0.8], [4.0, 0.0, 1.2]], dtype=mx.float32)

    scan = sensor.height_scan(roots)

    assert scan["distances"].shape == (2, 3)
    assert bool(mx.all(scan["hit_mask"]).item())
    assert bool(
        mx.allclose(
            scan["distances"],
            mx.array([[0.8, 0.8, 0.8], [1.2, 1.2, 1.2]], dtype=mx.float32),
        ).item()
    )


def test_height_scan_marks_out_of_bounds_rays_as_misses():
    """Rays that leave the terrain tile should clamp to max distance and clear the hit mask."""
    terrain = MacPlaneTerrain(1, env_spacing=4.0, tile_size=(0.6, 0.6), origin_height=0.0)
    sensor = MacPlaneRaycastSensor(terrain, offsets_xy=((0.0, 0.0), (0.5, 0.0)), max_distance=1.5)
    roots = mx.array([[0.0, 0.0, 0.5]], dtype=mx.float32)

    scan = sensor.height_scan(roots)

    assert bool(scan["hit_mask"][0, 0].item())
    assert not bool(scan["hit_mask"][0, 1].item())
    assert float(scan["distances"][0, 1].item()) == 1.5
