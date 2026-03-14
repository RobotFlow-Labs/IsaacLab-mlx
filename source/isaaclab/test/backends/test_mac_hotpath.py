# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the compiled MLX locomotion hot-path helpers."""

from __future__ import annotations

import math

import numpy as np

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim.hotpath import (  # noqa: E402
    HOTPATH_BACKEND,
    anymal_body_positions_hotpath,
    biped_support_metrics_hotpath,
    contact_update_hotpath,
    h1_body_positions_hotpath,
    quadruped_support_metrics_hotpath,
)


def test_contact_update_hotpath_matches_reference_math():
    body_pos_w = mx.array(
        [
            [[0.0, 0.0, 0.40], [0.2, 0.2, 0.005], [-0.2, 0.2, 0.010], [0.1, 0.0, 0.25]],
            [[0.0, 0.0, 0.015], [0.2, 0.2, 0.080], [-0.2, 0.2, 0.004], [0.1, 0.0, 0.010]],
        ],
        dtype=mx.float32,
    )
    body_vel_w = mx.zeros_like(body_pos_w)
    body_vel_w[:, :, 2] = mx.array([[0.0, -0.2, -0.1, 0.0], [-0.3, 0.0, -0.2, -0.3]], dtype=mx.float32)
    terrain_heights = mx.zeros((2, 4), dtype=mx.float32)
    previous_contact = mx.zeros((2, 4), dtype=mx.bool_)
    previous_history = mx.zeros((2, 3, 4, 3), dtype=mx.float32)
    previous_last_air_time = mx.array([[0.15, 0.20], [0.40, 0.50]], dtype=mx.float32)
    previous_air_time = mx.array([[0.30, 0.35], [0.60, 0.70]], dtype=mx.float32)
    foot_body_ids = mx.array([1, 2], dtype=mx.int32)

    current_contact, first_contact, history, last_air_time, air_time = contact_update_hotpath(
        body_pos_w,
        body_vel_w,
        terrain_heights,
        previous_contact,
        previous_history,
        previous_last_air_time,
        previous_air_time,
        foot_body_ids,
        contact_margin=0.02,
        spring_stiffness=2000.0,
        damping=50.0,
        force_threshold=1.0,
        step_dt=0.05,
    )
    mx.eval(current_contact, first_contact, history, last_air_time, air_time)

    clearance = np.array(body_pos_w)[:, :, 2]
    penetration = np.maximum(0.02 - clearance, 0.0)
    closing_speed = np.maximum(-np.array(body_vel_w)[:, :, 2], 0.0)
    normal_force = np.where(penetration > 0.0, 2000.0 * penetration + 50.0 * closing_speed, 0.0)
    expected_contact = normal_force > 1.0
    expected_first_contact = expected_contact
    expected_history = np.zeros((2, 3, 4, 3), dtype=np.float32)
    expected_history[:, 0, :, 2] = normal_force.astype(np.float32)
    expected_last_air = np.where(expected_first_contact[:, [1, 2]], np.array(previous_air_time), np.array(previous_last_air_time))
    expected_air = np.where(expected_contact[:, [1, 2]], 0.0, np.array(previous_air_time) + 0.05)

    assert np.array_equal(np.array(current_contact), expected_contact)
    assert np.array_equal(np.array(first_contact), expected_first_contact)
    assert np.allclose(np.array(history), expected_history)
    assert np.allclose(np.array(last_air_time), expected_last_air)
    assert np.allclose(np.array(air_time), expected_air)


def test_quadruped_support_metrics_hotpath_matches_reference():
    contact_mask = mx.array([[True, False, True, True], [False, True, False, True]], dtype=mx.bool_)
    action_targets = mx.arange(24, dtype=mx.float32).reshape((2, 12))

    outputs = quadruped_support_metrics_hotpath(contact_mask, action_targets)
    mx.eval(*outputs)
    support_ratio, left_support, right_support, front_support, rear_support, left_actions, right_actions, front_actions, rear_actions = outputs

    support = np.array(contact_mask, dtype=np.float32)
    grouped_actions = np.array(action_targets).reshape((2, 4, 3))

    assert np.allclose(np.array(support_ratio), np.mean(support, axis=1))
    assert np.allclose(np.array(left_support), np.mean(support[:, [0, 2]], axis=1))
    assert np.allclose(np.array(right_support), np.mean(support[:, [1, 3]], axis=1))
    assert np.allclose(np.array(front_support), np.mean(support[:, [0, 1]], axis=1))
    assert np.allclose(np.array(rear_support), np.mean(support[:, [2, 3]], axis=1))
    assert np.allclose(np.array(left_actions), np.mean(grouped_actions[:, [0, 2], :], axis=(1, 2)))
    assert np.allclose(np.array(right_actions), np.mean(grouped_actions[:, [1, 3], :], axis=(1, 2)))
    assert np.allclose(np.array(front_actions), np.mean(grouped_actions[:, [0, 1], :], axis=(1, 2)))
    assert np.allclose(np.array(rear_actions), np.mean(grouped_actions[:, [2, 3], :], axis=(1, 2)))


def test_biped_support_metrics_hotpath_matches_reference():
    contact_mask = mx.array([[True, False], [False, True]], dtype=mx.bool_)
    action_targets = mx.arange(38, dtype=mx.float32).reshape((2, 19))

    support_ratio, left_support, right_support, left_actions, right_actions = biped_support_metrics_hotpath(
        contact_mask, action_targets
    )
    mx.eval(support_ratio, left_support, right_support, left_actions, right_actions)

    support = np.array(contact_mask, dtype=np.float32)
    actions = np.array(action_targets)

    assert np.allclose(np.array(support_ratio), np.mean(support, axis=1))
    assert np.allclose(np.array(left_support), support[:, 0])
    assert np.allclose(np.array(right_support), support[:, 1])
    assert np.allclose(np.array(left_actions), np.mean(actions[:, :5], axis=1))
    assert np.allclose(np.array(right_actions), np.mean(actions[:, 5:10], axis=1))


def test_anymal_body_positions_hotpath_returns_expected_shape_and_base_slot():
    root_pos_w = mx.array([[0.1, 0.2, 0.5], [1.0, -0.4, 0.6]], dtype=mx.float32)
    joint_pos = mx.zeros((2, 12), dtype=mx.float32)
    commands = mx.array([[0.4, 0.0, 0.1], [0.0, 0.2, -0.2]], dtype=mx.float32)
    gait_phase = mx.array([0.0, math.pi / 2.0], dtype=mx.float32)
    hip_offsets = mx.array([[0.32, 0.20], [0.32, -0.20], [-0.32, 0.20], [-0.32, -0.20]], dtype=mx.float32)
    gait_phase_offsets = mx.array([0.0, math.pi, math.pi, 0.0], dtype=mx.float32)

    body_pos = anymal_body_positions_hotpath(root_pos_w, joint_pos, commands, gait_phase, hip_offsets, gait_phase_offsets, 0.08)
    mx.eval(body_pos)

    assert body_pos.shape == (2, 9, 3)
    assert np.allclose(np.array(body_pos)[:, 0, :], np.array(root_pos_w))


def test_h1_body_positions_hotpath_returns_expected_shape_and_base_slot():
    root_pos_w = mx.array([[0.0, 0.0, 0.92], [1.2, 0.1, 0.95]], dtype=mx.float32)
    joint_pos = mx.zeros((2, 19), dtype=mx.float32)
    commands = mx.array([[0.5, 0.0, 0.1], [0.2, 0.1, -0.1]], dtype=mx.float32)
    gait_phase = mx.array([0.0, math.pi / 3.0], dtype=mx.float32)
    hip_offsets = mx.array([[0.0, 0.11], [0.0, -0.11]], dtype=mx.float32)
    gait_phase_offsets = mx.array([0.0, math.pi], dtype=mx.float32)

    body_pos = h1_body_positions_hotpath(root_pos_w, joint_pos, commands, gait_phase, hip_offsets, gait_phase_offsets, 0.06)
    mx.eval(body_pos)

    assert body_pos.shape == (2, 5, 3)
    assert np.allclose(np.array(body_pos)[:, 0, :], np.array(root_pos_w))


def test_hotpath_backend_label_is_stable():
    assert HOTPATH_BACKEND == "mlx-compiled"
