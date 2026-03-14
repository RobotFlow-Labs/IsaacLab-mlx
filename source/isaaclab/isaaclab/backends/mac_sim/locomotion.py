# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-oriented reward and termination utilities for mac-native locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from .contacts import BatchedContactSensorState
from .terrain import MacPlaneTerrain


def track_linear_velocity_xy_exp(commands: mx.array, root_lin_vel_b: mx.array, std: float = 0.25) -> mx.array:
    """Reward tracking of commanded planar velocity using the upstream exponential form."""
    error = mx.sum(mx.square(commands[:, :2] - root_lin_vel_b[:, :2]), axis=1)
    return mx.exp(-error / std)


def track_yaw_rate_z_exp(commands: mx.array, root_ang_vel_b: mx.array, std: float = 0.25) -> mx.array:
    """Reward tracking of commanded yaw rate using the upstream exponential form."""
    error = mx.square(commands[:, 2] - root_ang_vel_b[:, 2])
    return mx.exp(-error / std)


def action_rate_l2(actions: mx.array, previous_actions: mx.array) -> mx.array:
    """Penalty on action delta magnitude."""
    return mx.sum(mx.square(actions - previous_actions), axis=1)


def flat_orientation_l2(projected_gravity_b: mx.array) -> mx.array:
    """Penalty on lateral gravity components for a nominally upright base."""
    return mx.sum(mx.square(projected_gravity_b[:, :2]), axis=1)


def feet_air_time_reward(
    contact_state: BatchedContactSensorState,
    commands: mx.array,
    *,
    baseline: float = 0.5,
    min_planar_command: float = 0.1,
) -> mx.array:
    """Reward first contact after a useful air-time duration when the robot is commanded to move."""
    first_contact = contact_state.compute_first_contact()
    reward = mx.sum((contact_state.last_air_time - baseline) * first_contact.astype(mx.float32), axis=1)
    moving = mx.linalg.norm(commands[:, :2], axis=1) > min_planar_command
    return reward * moving.astype(mx.float32)


def undesired_contacts(
    contact_state: BatchedContactSensorState,
    body_names: Sequence[str],
    *,
    threshold: float = 1.0,
) -> mx.array:
    """Count undesired body contacts above the selected threshold."""
    return mx.sum(contact_state.force_magnitudes(body_names) > threshold, axis=1).astype(mx.float32)


def base_contact_termination(
    contact_state: BatchedContactSensorState,
    body_names: Sequence[str],
    *,
    threshold: float = 1.0,
) -> mx.array:
    """Terminate when the selected base bodies register a strong terrain contact."""
    return contact_state.any_contact(body_names, threshold=threshold)


def terrain_out_of_bounds(terrain: MacPlaneTerrain, root_pos_w: mx.array, *, buffer: float = 0.5) -> mx.array:
    """Terminate when the root body exits the current terrain tile."""
    return terrain.out_of_bounds(root_pos_w, buffer=buffer)
