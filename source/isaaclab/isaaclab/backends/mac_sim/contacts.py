# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact approximations for the first mac-native locomotion task slice."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

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

        num_bodies = len(self.body_names)
        self.contact_mask = mx.zeros((num_envs, num_bodies), dtype=mx.bool_)
        self.first_contact = mx.zeros((num_envs, num_bodies), dtype=mx.bool_)
        self.net_forces_w_history = mx.zeros((num_envs, history_length, num_bodies, 3), dtype=mx.float32)
        self.last_air_time = mx.zeros((num_envs, len(self.foot_body_ids)), dtype=mx.float32)
        self._air_time = mx.zeros((num_envs, len(self.foot_body_ids)), dtype=mx.float32)

    def resolve_body_ids(self, body_names: Sequence[str] | None = None) -> tuple[int, ...]:
        """Resolve body names into deterministic index tuples."""
        if body_names is None:
            return tuple(range(len(self.body_names)))
        return tuple(self.body_name_to_index[name] for name in body_names)

    def update(self, body_pos_w: Any, body_vel_w: Any, terrain: MacPlaneTerrain) -> mx.array:
        """Update contact masks, contact forces, and air-time buffers."""
        body_pos_w = mx.array(body_pos_w, dtype=mx.float32).reshape((self.num_envs, len(self.body_names), 3))
        body_vel_w = mx.array(body_vel_w, dtype=mx.float32).reshape((self.num_envs, len(self.body_names), 3))

        terrain_heights = terrain.sample_heights(body_pos_w.reshape((-1, 3))).reshape((self.num_envs, len(self.body_names)))
        clearance = body_pos_w[:, :, 2] - terrain_heights
        penetration = mx.maximum(self.contact_margin - clearance, 0.0)
        closing_speed = mx.maximum(-body_vel_w[:, :, 2], 0.0)
        normal_force = mx.where(
            penetration > 0.0,
            self.spring_stiffness * penetration + self.damping * closing_speed,
            0.0,
        )
        forces_w = mx.stack(
            [
                mx.zeros_like(normal_force),
                mx.zeros_like(normal_force),
                normal_force,
            ],
            axis=-1,
        )
        current_contact = normal_force > self.force_threshold
        previous_contact = self.contact_mask
        self.first_contact = current_contact & ~previous_contact
        self.contact_mask = current_contact
        self.net_forces_w_history = mx.concatenate(
            [forces_w[:, None, :, :], self.net_forces_w_history[:, :-1, :, :]],
            axis=1,
        )

        if self.foot_body_ids:
            foot_ids = list(self.foot_body_ids)
            foot_contact = current_contact[:, foot_ids]
            foot_first_contact = self.first_contact[:, foot_ids]
            self.last_air_time = mx.where(foot_first_contact, self._air_time, self.last_air_time)
            self._air_time = mx.where(foot_contact, 0.0, self._air_time + self.step_dt)
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
        }
