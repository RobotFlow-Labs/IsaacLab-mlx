# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain primitives for the mac-native locomotion substrate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from .state_primitives import EnvironmentOriginGrid, env_ids_to_array


class MacPlaneTerrain:
    """Per-environment flat terrain tiles used for the first locomotion task bring-up."""

    def __init__(
        self,
        num_envs: int,
        *,
        env_spacing: float,
        tile_size: tuple[float, float] = (4.0, 4.0),
        border_width: float = 0.0,
        origin_height: float = 0.0,
    ):
        self.num_envs = num_envs
        self.tile_size = tile_size
        self.border_width = border_width
        self.origin_height = origin_height
        self.origin_grid = EnvironmentOriginGrid(num_envs, env_spacing)

    @property
    def env_origins(self) -> mx.array:
        return self.origin_grid.origins

    def spawn_positions(self, env_ids: Sequence[int], offset: Any = (0.0, 0.0, 0.0)) -> mx.array:
        """Return environment-local spawn positions offset from the tile origin."""
        positions = self.origin_grid.positions_with_offset(env_ids, offset)
        positions[:, 2] = positions[:, 2] + self.origin_height
        return positions

    def sample_heights(self, positions_w: Any) -> mx.array:
        """Return terrain height at world positions."""
        positions = mx.array(positions_w, dtype=mx.float32).reshape((-1, 3))
        return mx.full((positions.shape[0],), self.origin_height, dtype=mx.float32)

    def surface_normals(self, positions_w: Any) -> mx.array:
        """Return terrain surface normals at world positions."""
        positions = mx.array(positions_w, dtype=mx.float32).reshape((-1, 3))
        normal = mx.array([0.0, 0.0, 1.0], dtype=mx.float32)
        return mx.broadcast_to(normal.reshape((1, 3)), positions.shape)

    def signed_height(self, positions_w: Any) -> mx.array:
        """Return signed height relative to the plane."""
        positions = mx.array(positions_w, dtype=mx.float32).reshape((-1, 3))
        return positions[:, 2] - self.sample_heights(positions)

    def out_of_bounds(
        self,
        positions_w: Any,
        *,
        env_ids: Sequence[int] | None = None,
        buffer: float = 0.0,
    ) -> mx.array:
        """Return whether any tracked positions leave their terrain tile."""
        positions = mx.array(positions_w, dtype=mx.float32)
        if positions.ndim == 2:
            positions = positions.reshape((positions.shape[0], 1, 3))
        if positions.ndim != 3:
            raise ValueError("Expected positions with shape (num_envs, num_bodies, 3) or (num_envs, 3).")

        rows = positions.shape[0]
        if env_ids is None:
            if rows != self.num_envs:
                raise ValueError("env_ids must be provided when positions do not cover every environment.")
            ids = mx.arange(self.num_envs, dtype=mx.int32)
        else:
            ids = env_ids_to_array(env_ids, self.num_envs)

        origins = self.env_origins[ids][:, None, :2]
        rel_xy = positions[:, :, :2] - origins
        max_x = self.tile_size[0] * 0.5 + self.border_width - buffer
        max_y = self.tile_size[1] * 0.5 + self.border_width - buffer
        exceeds = (mx.abs(rel_xy[:, :, 0]) > max_x) | (mx.abs(rel_xy[:, :, 1]) > max_y)
        return mx.any(exceeds, axis=1)

    def state_dict(self) -> dict[str, Any]:
        """Return a compact diagnostics payload."""
        return {
            "type": "plane",
            "num_envs": self.num_envs,
            "tile_size": list(self.tile_size),
            "border_width": self.border_width,
            "origin_height": self.origin_height,
            "env_spacing": self.origin_grid.spacing,
        }
