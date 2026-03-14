# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain primitives for the mac-native locomotion substrate."""

from __future__ import annotations

from collections.abc import Sequence
import math
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

    def sample_heights(self, positions_w: Any, *, env_ids: Sequence[int] | None = None) -> mx.array:
        """Return terrain height at world positions."""
        del env_ids
        positions = mx.array(positions_w, dtype=mx.float32).reshape((-1, 3))
        return mx.full((positions.shape[0],), self.origin_height, dtype=mx.float32)

    def surface_normals(self, positions_w: Any, *, env_ids: Sequence[int] | None = None) -> mx.array:
        """Return terrain surface normals at world positions."""
        del env_ids
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


class MacWaveTerrain(MacPlaneTerrain):
    """Procedural sinusoidal terrain used for the first raycast-driven locomotion slice."""

    def __init__(
        self,
        num_envs: int,
        *,
        env_spacing: float,
        tile_size: tuple[float, float] = (5.0, 5.0),
        border_width: float = 0.0,
        origin_height: float = 0.0,
        amplitude: float = 0.05,
        wavelength: tuple[float, float] = (1.4, 1.0),
    ):
        super().__init__(
            num_envs,
            env_spacing=env_spacing,
            tile_size=tile_size,
            border_width=border_width,
            origin_height=origin_height,
        )
        self.amplitude = amplitude
        self.wavelength = wavelength

    def _relative_xy(
        self,
        positions_w: Any,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[mx.array, mx.array]:
        positions = mx.array(positions_w, dtype=mx.float32).reshape((-1, 3))
        rows = positions.shape[0]
        if env_ids is None:
            if rows != self.num_envs:
                raise ValueError("env_ids must be provided when positions do not cover every environment.")
            ids = mx.arange(self.num_envs, dtype=mx.int32)
        else:
            ids = env_ids_to_array(env_ids, self.num_envs)
            if len(ids) != rows:
                raise ValueError("env_ids length must match the number of queried positions.")
        origins = self.env_origins[ids]
        rel_xy = positions[:, :2] - origins[:, :2]
        return rel_xy, ids

    def sample_heights(self, positions_w: Any, *, env_ids: Sequence[int] | None = None) -> mx.array:
        """Return wave terrain heights at world positions."""

        rel_xy, _ = self._relative_xy(positions_w, env_ids=env_ids)
        phase_x = rel_xy[:, 0] * (2.0 * math.pi / self.wavelength[0])
        phase_y = rel_xy[:, 1] * (2.0 * math.pi / self.wavelength[1])
        heights = self.origin_height + self.amplitude * mx.sin(phase_x) * mx.cos(phase_y)
        return heights.astype(mx.float32)

    def surface_normals(self, positions_w: Any, *, env_ids: Sequence[int] | None = None) -> mx.array:
        """Return analytic wave terrain normals at world positions."""

        rel_xy, _ = self._relative_xy(positions_w, env_ids=env_ids)
        phase_x = rel_xy[:, 0] * (2.0 * math.pi / self.wavelength[0])
        phase_y = rel_xy[:, 1] * (2.0 * math.pi / self.wavelength[1])
        dh_dx = self.amplitude * (2.0 * math.pi / self.wavelength[0]) * mx.cos(phase_x) * mx.cos(phase_y)
        dh_dy = -self.amplitude * (2.0 * math.pi / self.wavelength[1]) * mx.sin(phase_x) * mx.sin(phase_y)
        normals = mx.stack((-dh_dx, -dh_dy, mx.ones_like(dh_dx)), axis=-1)
        return normals / mx.linalg.norm(normals, axis=-1, keepdims=True)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update(
            {
                "type": "wave",
                "amplitude": self.amplitude,
                "wavelength": list(self.wavelength),
            }
        )
        return payload
