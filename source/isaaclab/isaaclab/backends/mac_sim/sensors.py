# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native analytic sensor primitives for the first MLX sensor slice."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from .terrain import MacPlaneTerrain

DEFAULT_HEIGHT_SCAN_OFFSETS = (
    (-0.35, -0.35),
    (-0.35, 0.0),
    (-0.35, 0.35),
    (0.0, -0.35),
    (0.0, 0.0),
    (0.0, 0.35),
    (0.35, -0.35),
    (0.35, 0.0),
    (0.35, 0.35),
)


class MacPlaneRaycastSensor:
    """Analytic downward raycast and height-scan sensor for flat plane terrain."""

    def __init__(
        self,
        terrain: MacPlaneTerrain,
        *,
        offsets_xy: Sequence[tuple[float, float]] = DEFAULT_HEIGHT_SCAN_OFFSETS,
        max_distance: float = 2.0,
    ):
        self.terrain = terrain
        self.max_distance = max_distance
        self.offsets_xy = mx.array(offsets_xy, dtype=mx.float32).reshape((-1, 2))

    @property
    def scan_dim(self) -> int:
        return int(self.offsets_xy.shape[0])

    def raycast(
        self,
        origins_w: Any,
        *,
        directions_w: Any | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, mx.array]:
        """Cast analytic rays against the plane terrain."""

        origins = mx.array(origins_w, dtype=mx.float32).reshape((-1, 3))
        if directions_w is None:
            directions = mx.broadcast_to(mx.array([0.0, 0.0, -1.0], dtype=mx.float32), origins.shape)
        else:
            directions = mx.array(directions_w, dtype=mx.float32).reshape(origins.shape)

        downward = directions[:, 2] < -1e-6
        plane_height = self.terrain.sample_heights(origins)
        distances = (plane_height - origins[:, 2]) / directions[:, 2]
        hit_points = origins + directions * distances[:, None]
        normals = self.terrain.surface_normals(hit_points)
        within_distance = (distances >= 0.0) & (distances <= self.max_distance)
        within_bounds = ~self.terrain.out_of_bounds(hit_points, env_ids=env_ids)
        hits = downward & within_distance & within_bounds
        clamped_distances = mx.where(hits, distances, self.max_distance)
        normals = mx.where(hits[:, None], normals, 0.0)
        hit_points = mx.where(hits[:, None], hit_points, origins + directions * self.max_distance)
        return {
            "hit_points_w": hit_points,
            "hit_distances": clamped_distances.astype(mx.float32),
            "hit_normals_w": normals.astype(mx.float32),
            "hit_mask": hits.astype(mx.bool_),
        }

    def height_scan(self, root_pos_w: Any, *, env_ids: Sequence[int] | None = None) -> dict[str, mx.array]:
        """Cast a fixed downward ray grid around each root position."""

        roots = mx.array(root_pos_w, dtype=mx.float32).reshape((-1, 3))
        rows = roots.shape[0]
        offsets = mx.broadcast_to(self.offsets_xy[None, :, :], (rows, self.scan_dim, 2))
        origins = mx.broadcast_to(roots[:, None, :], (rows, self.scan_dim, 3))
        origins[:, :, :2] = origins[:, :, :2] + offsets
        flat_origins = origins.reshape((-1, 3))
        base_env_ids = list(range(rows)) if env_ids is None else list(env_ids)
        flat_env_ids = [env_id for env_id in base_env_ids for _ in range(self.scan_dim)]
        result = self.raycast(flat_origins, env_ids=flat_env_ids)
        return {
            "distances": result["hit_distances"].reshape((rows, self.scan_dim)),
            "hit_mask": result["hit_mask"].reshape((rows, self.scan_dim)),
            "hit_points_w": result["hit_points_w"].reshape((rows, self.scan_dim, 3)),
            "hit_normals_w": result["hit_normals_w"].reshape((rows, self.scan_dim, 3)),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return serializable sensor metadata for diagnostics and benchmarks."""

        return {
            "backend": "mac-sensors",
            "implementation": "analytic-plane-raycast",
            "scan_dim": self.scan_dim,
            "max_distance": self.max_distance,
        }
