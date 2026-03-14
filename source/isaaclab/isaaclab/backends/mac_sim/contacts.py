# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact approximations for the first mac-native locomotion task slice."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from .hotpath import HOTPATH_BACKEND, contact_update_hotpath
from .state_primitives import env_ids_to_array
from .terrain import MacPlaneTerrain


class BatchedContactSensorState:
    """A lightweight contact approximation buffer for flat-terrain locomotion bring-up."""

    def __init__(
        self,
        num_envs: int,
        body_names: Sequence[str],
        *,
        foot_body_names: Sequence[str] = (),
        step_dt: float,
        history_length: int = 3,
        contact_margin: float = 0.02,
        spring_stiffness: float = 2000.0,
        damping: float = 50.0,
        force_threshold: float = 1.0,
    ):
        self.num_envs = num_envs
        self.body_names = tuple(body_names)
        self.body_name_to_index = {name: index for index, name in enumerate(self.body_names)}
        self.foot_body_ids = tuple(self.body_name_to_index[name] for name in foot_body_names)
        self.step_dt = step_dt
        self.history_length = history_length
        self.contact_margin = contact_margin
        self.spring_stiffness = spring_stiffness
        self.damping = damping
        self.force_threshold = force_threshold
        self.hotpath_backend = HOTPATH_BACKEND

        num_bodies = len(self.body_names)
        self.contact_mask = mx.zeros((num_envs, num_bodies), dtype=mx.bool_)
        self.first_contact = mx.zeros((num_envs, num_bodies), dtype=mx.bool_)
        self.net_forces_w_history = mx.zeros((num_envs, history_length, num_bodies, 3), dtype=mx.float32)
        self.last_air_time = mx.zeros((num_envs, len(self.foot_body_ids)), dtype=mx.float32)
        self._air_time = mx.zeros((num_envs, len(self.foot_body_ids)), dtype=mx.float32)
        self._foot_body_ids_array = mx.array(self.foot_body_ids, dtype=mx.int32)

    def reset_envs(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset contact and air-time buffers for a subset of environments."""
        ids = env_ids_to_array(env_ids, self.num_envs)
        self.contact_mask[ids] = False
        self.first_contact[ids] = False
        self.net_forces_w_history[ids] = 0.0
        if self.foot_body_ids:
            self.last_air_time[ids] = 0.0
            self._air_time[ids] = 0.0

    def resolve_body_ids(self, body_names: Sequence[str] | None = None) -> tuple[int, ...]:
        """Resolve body names into deterministic index tuples."""
        if body_names is None:
            return tuple(range(len(self.body_names)))
        return tuple(self.body_name_to_index[name] for name in body_names)

    def update(
        self,
        body_pos_w: Any,
        body_vel_w: Any,
        terrain: MacPlaneTerrain,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> mx.array:
        """Update contact masks, contact forces, and air-time buffers."""
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(ids)
        body_pos_w = mx.array(body_pos_w, dtype=mx.float32).reshape((rows, len(self.body_names), 3))
        body_vel_w = mx.array(body_vel_w, dtype=mx.float32).reshape((rows, len(self.body_names), 3))

        terrain_heights = terrain.sample_heights(body_pos_w.reshape((-1, 3))).reshape((rows, len(self.body_names)))
        current_contact, first_contact, history, last_air_time, air_time = contact_update_hotpath(
            body_pos_w,
            body_vel_w,
            terrain_heights,
            self.contact_mask[ids],
            self.net_forces_w_history[ids],
            self.last_air_time[ids],
            self._air_time[ids],
            self._foot_body_ids_array,
            contact_margin=self.contact_margin,
            spring_stiffness=self.spring_stiffness,
            damping=self.damping,
            force_threshold=self.force_threshold,
            step_dt=self.step_dt,
        )
        self.first_contact[ids] = first_contact
        self.contact_mask[ids] = current_contact
        self.net_forces_w_history[ids] = history
        if self.foot_body_ids:
            self.last_air_time[ids] = last_air_time
            self._air_time[ids] = air_time
        return current_contact

    def compute_first_contact(self, step_dt: float | None = None) -> mx.array:
        """Return the current first-contact mask for tracked feet."""
        del step_dt
        if self.foot_body_ids:
            return self.first_contact[:, list(self.foot_body_ids)]
        return self.first_contact

    def force_magnitudes(self, body_names: Sequence[str] | None = None) -> mx.array:
        """Return latest contact force magnitudes for selected bodies."""
        body_ids = list(self.resolve_body_ids(body_names))
        forces = self.net_forces_w_history[:, 0, body_ids, :]
        return mx.linalg.norm(forces, axis=-1)

    def any_contact(self, body_names: Sequence[str], *, threshold: float | None = None) -> mx.array:
        """Return whether any selected body exceeds a contact threshold."""
        threshold = self.force_threshold if threshold is None else threshold
        return mx.any(self.force_magnitudes(body_names) > threshold, axis=1)

    def state_dict(self) -> dict[str, Any]:
        """Return a compact diagnostics payload."""
        return {
            "tracked_bodies": list(self.body_names),
            "foot_body_ids": list(self.foot_body_ids),
            "history_length": self.history_length,
            "contact_margin": self.contact_margin,
            "force_threshold": self.force_threshold,
            "hotpath_backend": self.hotpath_backend,
        }
