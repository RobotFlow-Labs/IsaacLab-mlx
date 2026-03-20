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
    _locomotion_root_step_impl,
    anymal_body_positions_hotpath,
    anymal_leg_extension_hotpath,
    biped_support_metrics_hotpath,
    contact_update_hotpath,
    franka_cabinet_step_hotpath,
    franka_end_effector_position_hotpath,
    franka_lift_object_step_hotpath,
    franka_stack_object_step_hotpath,
    franka_stack_rgb_step_hotpath,
    get_anymal_leg_extension_hotpath_backend,
    get_contact_hotpath_backend,
    get_franka_cabinet_hotpath_backend,
    get_franka_hotpath_backend,
    get_franka_lift_hotpath_backend,
    get_franka_stack_hotpath_backend,
    get_franka_stack_rgb_hotpath_backend,
    get_locomotion_hotpath_backend,
    h1_body_positions_hotpath,
    h1_leg_extension_hotpath,
    get_h1_leg_extension_hotpath_backend,
    locomotion_root_step_hotpath,
    prime_contact_state,
    quadruped_support_metrics_hotpath,
)
from isaaclab.backends.mac_sim import BatchedContactSensorState, MacPlaneTerrain  # noqa: E402


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


def test_anymal_leg_extension_hotpath_matches_reference_math():
    joint_pos = mx.array(
        [
            [0.0, -0.4, 0.75, 0.1, -0.2, 0.65, -0.1, 0.25, -0.15, -0.3, 0.55, 0.05],
            [0.1, -0.3, 0.55, -0.2, 0.0, 0.45, 0.2, -0.1, 0.15, 0.3, -0.25, 0.35],
        ],
        dtype=mx.float32,
    )

    extension = anymal_leg_extension_hotpath(joint_pos)
    mx.eval(extension)

    joint_np = np.array(joint_pos).reshape((2, 4, 3))
    expected = 0.20 + 0.16 * np.cos(joint_np[:, :, 1]) + 0.18 * np.cos(joint_np[:, :, 1] + joint_np[:, :, 2])
    expected = np.clip(expected, 0.22, 0.62).astype(np.float32)

    assert extension.shape == (2, 4)
    assert np.allclose(np.array(extension), expected)


def test_h1_body_positions_hotpath_returns_expected_shape_and_base_slot():
    root_pos_w = mx.array([[0.0, 0.0, 0.92], [1.2, 0.1, 0.95]], dtype=mx.float32)
    joint_pos = mx.array(
        [
            [0.02, -0.10, 0.45, -0.20, 0.08, 0.12, -0.06, 0.14, 0.18, -0.04, 0.05, 0.01, -0.02, 0.03, 0.04, -0.05, 0.06, -0.07, 0.08],
            [-0.03, 0.15, 0.35, 0.05, -0.09, -0.11, 0.07, -0.13, 0.09, 0.12, -0.04, 0.02, 0.06, -0.08, 0.10, 0.11, -0.12, 0.13, -0.14],
        ],
        dtype=mx.float32,
    )
    commands = mx.array([[0.5, 0.0, 0.1], [0.2, 0.1, -0.1]], dtype=mx.float32)
    gait_phase = mx.array([0.0, math.pi / 3.0], dtype=mx.float32)
    hip_offsets = mx.array([[0.0, 0.11], [0.0, -0.11]], dtype=mx.float32)
    gait_phase_offsets = mx.array([0.0, math.pi], dtype=mx.float32)

    body_pos = h1_body_positions_hotpath(root_pos_w, joint_pos, commands, gait_phase, hip_offsets, gait_phase_offsets, 0.06)
    mx.eval(body_pos)

    assert body_pos.shape == (2, 5, 3)
    root_np = np.array(root_pos_w)
    joint_np = np.array(joint_pos)[:, :10].reshape((2, 2, 5))
    commands_np = np.array(commands)
    gait_phase_np = np.array(gait_phase)[:, None]
    hip_offsets_np = np.array(hip_offsets).reshape((1, 2, 2))
    gait_phase_offsets_np = np.array(gait_phase_offsets).reshape((1, 2))
    hip_yaw = joint_np[:, :, 0]
    hip_roll = joint_np[:, :, 1]
    hip_pitch = joint_np[:, :, 2]
    knee = joint_np[:, :, 3]
    ankle = joint_np[:, :, 4]
    extension = 0.40 + 0.20 * np.cos(hip_pitch + 0.20) + 0.26 * np.cos(hip_pitch + knee - 0.10) + 0.08 * np.cos(
        hip_pitch + knee + ankle
    )
    extension = np.clip(extension, 0.58, 0.98)
    command_speed = np.linalg.norm(commands_np[:, :2], axis=1, keepdims=True)
    phase = gait_phase_np + gait_phase_offsets_np
    swing = np.maximum(np.sin(phase), 0.0) * (0.18 + 0.65 * command_speed)
    root_xy = root_np[:, None, :2]
    root_z = root_np[:, 2:3]
    step_x = 0.18 * commands_np[:, 0:1] * np.cos(phase)
    step_y = 0.06 * commands_np[:, 1:2] * np.sin(phase)
    foot_xy = root_xy + hip_offsets_np
    foot_xy_x = foot_xy[:, :, 0] + step_x - 0.04 * np.sin(hip_pitch)
    foot_xy_y = foot_xy[:, :, 1] + step_y + 0.03 * hip_roll + 0.02 * hip_yaw
    foot_xy = np.stack([foot_xy_x, foot_xy_y], axis=-1)
    foot_z = root_z - extension + 0.06 * swing - 0.03 * np.tanh(ankle)
    foot_pos = np.concatenate([foot_xy, foot_z[:, :, None]], axis=-1)
    knee_xy = root_xy + 0.5 * hip_offsets_np
    knee_xy_x = knee_xy[:, :, 0] + 0.5 * step_x - 0.02 * np.sin(hip_pitch)
    knee_xy_y = knee_xy[:, :, 1] + 0.015 * hip_roll
    knee_xy = np.stack([knee_xy_x, knee_xy_y], axis=-1)
    knee_z = root_z - 0.42 - 0.18 * np.tanh(hip_pitch) - 0.08 * np.tanh(knee)
    knee_pos = np.concatenate([knee_xy, knee_z[:, :, None]], axis=-1)
    expected = np.concatenate([root_np[:, None, :], foot_pos, knee_pos], axis=1).astype(np.float32)

    assert np.allclose(np.array(body_pos), expected)


def test_h1_leg_extension_hotpath_matches_reference_math():
    joint_pos = mx.array(
        [
            [0.05, -0.25, 0.70, -0.15, 0.05, 0.12, 0.10, -0.08, 0.18, 0.22, -0.30, 0.04, 0.12, -0.07, 0.20, -0.04, 0.15, 0.08, -0.02, 0.11],
            [0.00, -0.30, 0.60, 0.08, -0.10, 0.18, -0.05, 0.12, -0.16, 0.06, 0.20, -0.02, 0.14, 0.03, -0.12, 0.10, -0.18, 0.05, 0.09, -0.06],
        ],
        dtype=mx.float32,
    )

    extension = h1_leg_extension_hotpath(joint_pos)
    mx.eval(extension)

    joint_np = np.array(joint_pos)[:, :10].reshape((2, 2, 5))
    hip_pitch = joint_np[:, :, 2]
    knee = joint_np[:, :, 3]
    ankle = joint_np[:, :, 4]
    expected = 0.40 + 0.20 * np.cos(hip_pitch + 0.20) + 0.26 * np.cos(hip_pitch + knee - 0.10) + 0.08 * np.cos(
        hip_pitch + knee + ankle
    )
    expected = np.clip(expected, 0.58, 0.98).astype(np.float32)

    assert extension.shape == (2, 2)
    assert np.allclose(np.array(extension), expected)


def test_hotpath_backend_label_is_stable():
    assert HOTPATH_BACKEND == "mlx-compiled"
    assert get_anymal_leg_extension_hotpath_backend() in {"mlx-compiled", "mlx-metal-anymal-leg-extension"}
    assert get_contact_hotpath_backend() in {"mlx-compiled", "mlx-metal-contact"}
    assert get_franka_cabinet_hotpath_backend() in {"mlx-compiled", "mlx-metal-franka-cabinet"}
    assert get_franka_hotpath_backend() in {"mlx-compiled", "mlx-metal-ee"}
    assert get_franka_lift_hotpath_backend() in {"mlx-compiled", "mlx-metal-franka-lift"}
    assert get_franka_stack_hotpath_backend() in {"mlx-compiled", "mlx-metal-franka-stack"}
    assert get_franka_stack_rgb_hotpath_backend() in {"mlx-compiled", "mlx-metal-franka-stack-rgb"}
    assert get_locomotion_hotpath_backend() in {"mlx-compiled", "mlx-metal-root-step"}
    assert get_h1_leg_extension_hotpath_backend() in {"mlx-compiled", "mlx-metal-h1-leg-extension"}


def test_franka_end_effector_hotpath_matches_reference_math():
    joint_pos = mx.array(
        [
            [0.0, -0.55, 0.0, -2.1, 0.0, 1.65, 0.75],
            [0.1, -0.45, 0.05, -1.9, -0.1, 1.4, 0.5],
        ],
        dtype=mx.float32,
    )

    ee_pos = franka_end_effector_position_hotpath(joint_pos)
    mx.eval(ee_pos)

    joint_np = np.array(joint_pos)
    q0, q1, q2, q3, q4, q5, q6 = [joint_np[:, idx] for idx in range(7)]
    expected = np.stack(
        (
            0.28
            + 0.18 * np.cos(q0) * np.cos(q1)
            + 0.14 * np.cos(q0) * np.cos(q1 + q2)
            + 0.08 * np.cos(q0) * np.cos(q1 + q2 + 0.5 * q3)
            - 0.02 * np.sin(q4),
            0.20 * np.sin(q0) + 0.07 * np.sin(q0 + 0.5 * q3) + 0.03 * np.tanh(q5) + 0.015 * np.sin(q6),
            0.28 + 0.18 * np.sin(-q1) + 0.13 * np.sin(-(q1 + q2)) + 0.07 * np.sin(-(q1 + q2 + 0.5 * q3)),
        ),
        axis=-1,
    ).astype(np.float32)

    assert np.allclose(np.array(ee_pos), expected)


def test_franka_lift_object_hotpath_matches_reference_math():
    cube_pos_w = mx.array([[0.55, 0.02, 0.08], [0.62, -0.03, 0.04]], dtype=mx.float32)
    ee_pos_w = mx.array([[0.56, 0.01, 0.14], [0.70, -0.02, 0.16]], dtype=mx.float32)
    gripper_joint_pos = mx.array([0.04, 0.04], dtype=mx.float32)
    gripper_action = mx.array([-1.0, 1.0], dtype=mx.float32)
    grasped = mx.array([False, True], dtype=mx.bool_)

    gripper_target, gripper_velocity, next_grasped, next_cube_pos = franka_lift_object_step_hotpath(
        cube_pos_w,
        ee_pos_w,
        gripper_joint_pos,
        gripper_action,
        grasped,
        0.02,
        0.0,
        0.08,
        0.03,
        0.07,
        0.055,
        0.04,
    )
    mx.eval(gripper_target, gripper_velocity, next_grasped, next_cube_pos)

    cube_np = np.array(cube_pos_w)
    ee_np = np.array(ee_pos_w)
    action_np = np.array(gripper_action)
    grasped_np = np.array(grasped)
    target_np = np.where(action_np < 0.0, 0.0, 0.08).astype(np.float32)
    velocity_np = (target_np - np.array(gripper_joint_pos)) / 0.02
    dist_np = np.linalg.norm(cube_np - ee_np, axis=1)
    closed_np = target_np <= 0.03
    can_grasp_np = dist_np <= 0.07
    next_grasped_np = (grasped_np | (can_grasp_np & closed_np)) & closed_np
    attached_np = ee_np + np.array([0.0, 0.0, -0.055], dtype=np.float32)
    resting_np = np.stack(
        (
            cube_np[:, 0],
            cube_np[:, 1],
            np.maximum(0.04, cube_np[:, 2] - 0.02 * 0.35),
        ),
        axis=-1,
    ).astype(np.float32)
    next_cube_np = np.where(next_grasped_np[:, None], attached_np, resting_np)

    assert np.allclose(np.array(gripper_target), target_np)
    assert np.allclose(np.array(gripper_velocity), velocity_np)
    assert np.array_equal(np.array(next_grasped), next_grasped_np)
    assert np.allclose(np.array(next_cube_pos), next_cube_np)


def test_franka_stack_object_hotpath_matches_reference_math():
    cube_pos_w = mx.array([[0.55, 0.10, 0.08], [0.60, -0.02, 0.08]], dtype=mx.float32)
    support_cube_pos_w = mx.array([[0.50, 0.02, 0.04], [0.58, -0.02, 0.04]], dtype=mx.float32)
    ee_pos_w = mx.array([[0.50, 0.02, 0.095], [0.66, -0.02, 0.18]], dtype=mx.float32)
    gripper_joint_pos = mx.array([0.0, 0.0], dtype=mx.float32)
    gripper_action = mx.array([1.0, -1.0], dtype=mx.float32)
    grasped = mx.array([True, False], dtype=mx.bool_)
    stacked = mx.array([False, False], dtype=mx.bool_)

    gripper_target, gripper_velocity, next_grasped, next_stacked, next_cube_pos = franka_stack_object_step_hotpath(
        cube_pos_w,
        support_cube_pos_w,
        ee_pos_w,
        gripper_joint_pos,
        gripper_action,
        grasped,
        stacked,
        0.02,
        0.0,
        0.08,
        0.03,
        0.05,
        0.07,
        0.055,
        0.04,
        0.04,
        0.04,
        0.03,
    )
    mx.eval(gripper_target, gripper_velocity, next_grasped, next_stacked, next_cube_pos)

    cube_np = np.array(cube_pos_w)
    support_np = np.array(support_cube_pos_w)
    ee_np = np.array(ee_pos_w)
    target_np = np.where(np.array(gripper_action) < 0.0, 0.0, 0.08).astype(np.float32)
    velocity_np = (target_np - np.array(gripper_joint_pos)) / 0.02
    dist_np = np.linalg.norm(cube_np - ee_np, axis=1)
    gripper_closed_np = target_np <= 0.03
    gripper_open_np = target_np >= 0.05
    can_grasp_np = dist_np <= 0.07
    attached_np = ee_np + np.array([0.0, 0.0, -0.055], dtype=np.float32)
    stack_target_np = support_np + np.array([0.0, 0.0, 0.04], dtype=np.float32)
    release_np = np.where(np.array(grasped)[:, None], attached_np, cube_np)
    stack_xy_error_np = np.linalg.norm(release_np[:, :2] - stack_target_np[:, :2], axis=1)
    stack_z_error_np = np.abs(release_np[:, 2] - stack_target_np[:, 2])
    newly_stacked_np = np.array(grasped) & gripper_open_np & (stack_xy_error_np <= 0.04) & (stack_z_error_np <= 0.03)
    next_stacked_np = np.array(stacked) | newly_stacked_np
    next_grasped_np = (np.array(grasped) | (can_grasp_np & gripper_closed_np)) & gripper_closed_np & ~np.array(stacked)
    resting_np = np.stack(
        (
            release_np[:, 0],
            release_np[:, 1],
            np.maximum(0.04, release_np[:, 2] - 0.02 * 0.35),
        ),
        axis=-1,
    ).astype(np.float32)
    next_cube_np = np.where(
        next_stacked_np[:, None],
        stack_target_np,
        np.where(next_grasped_np[:, None], attached_np, resting_np),
    )
    next_grasped_np = next_grasped_np & ~next_stacked_np

    assert np.allclose(np.array(gripper_target), target_np)
    assert np.allclose(np.array(gripper_velocity), velocity_np)
    assert np.array_equal(np.array(next_grasped), next_grasped_np)
    assert np.array_equal(np.array(next_stacked), next_stacked_np)
    assert np.allclose(np.array(next_cube_pos), next_cube_np)


def test_franka_stack_release_miss_keeps_current_release_pose():
    cube_pos_w = mx.array([[0.35, -0.10, 0.04]], dtype=mx.float32)
    support_cube_pos_w = mx.array([[0.45, 0.00, 0.04]], dtype=mx.float32)
    ee_pos_w = mx.array([[0.52, 0.08, 0.18]], dtype=mx.float32)
    gripper_joint_pos = mx.array([0.0], dtype=mx.float32)
    gripper_action = mx.array([1.0], dtype=mx.float32)
    grasped = mx.array([True], dtype=mx.bool_)
    stacked = mx.array([False], dtype=mx.bool_)

    _, _, next_grasped, next_stacked, next_cube_pos = franka_stack_object_step_hotpath(
        cube_pos_w,
        support_cube_pos_w,
        ee_pos_w,
        gripper_joint_pos,
        gripper_action,
        grasped,
        stacked,
        0.1,
        0.0,
        0.08,
        0.03,
        0.05,
        0.07,
        0.055,
        0.04,
        0.04,
        0.04,
        0.03,
    )
    mx.eval(next_grasped, next_stacked, next_cube_pos)

    expected_release_pose = np.array([[0.52, 0.08, 0.125]], dtype=np.float32)
    expected_resting_pose = np.array([[0.52, 0.08, 0.09]], dtype=np.float32)

    assert np.array_equal(np.array(next_grasped), np.array([False]))
    assert np.array_equal(np.array(next_stacked), np.array([False]))
    assert np.allclose(np.array(next_cube_pos), expected_resting_pose)
    assert np.allclose(np.array(next_cube_pos)[:, :2], expected_release_pose[:, :2])


def test_franka_cabinet_step_hotpath_matches_reference_math():
    handle_anchor_pos_w = mx.array([[0.48, 0.01, 0.20], [0.52, -0.02, 0.18]], dtype=mx.float32)
    ee_pos_w = mx.array([[0.68, 0.01, 0.20], [0.55, -0.02, 0.18]], dtype=mx.float32)
    gripper_joint_pos = mx.array([0.04, 0.02], dtype=mx.float32)
    gripper_action = mx.array([-1.0, 1.0], dtype=mx.float32)
    grasped = mx.array([True, False], dtype=mx.bool_)
    opened = mx.array([False, False], dtype=mx.bool_)
    drawer_open_amount = mx.array([0.0, 0.03], dtype=mx.float32)

    gripper_target, gripper_velocity, next_grasped, next_opened, next_drawer_open_amount, next_handle_pos = (
        franka_cabinet_step_hotpath(
            handle_anchor_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            grasped,
            opened,
            drawer_open_amount,
            0.02,
            0.0,
            0.08,
            0.03,
            0.065,
            0.24,
            0.18,
        )
    )
    mx.eval(gripper_target, gripper_velocity, next_grasped, next_opened, next_drawer_open_amount, next_handle_pos)

    anchor_np = np.array(handle_anchor_pos_w)
    ee_np = np.array(ee_pos_w)
    target_np = np.where(np.array(gripper_action) < 0.0, 0.0, 0.08).astype(np.float32)
    velocity_np = (target_np - np.array(gripper_joint_pos)) / 0.02
    current_handle_np = anchor_np + np.stack(
        [np.array(drawer_open_amount), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)],
        axis=-1,
    )
    handle_dist_np = np.linalg.norm(current_handle_np - ee_np, axis=1)
    gripper_closed_np = target_np <= 0.03
    can_grasp_np = handle_dist_np <= 0.065
    next_grasped_np = (np.array(grasped) | (can_grasp_np & gripper_closed_np)) & gripper_closed_np & ~np.array(opened)
    pulled_open_np = np.clip(ee_np[:, 0] - anchor_np[:, 0], 0.0, 0.24).astype(np.float32)
    next_drawer_np = np.where(next_grasped_np, pulled_open_np, np.array(drawer_open_amount)).astype(np.float32)
    next_opened_np = np.array(opened) | (next_drawer_np >= 0.18)
    next_handle_np = anchor_np + np.stack(
        [next_drawer_np, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)],
        axis=-1,
    )
    next_grasped_np = next_grasped_np & ~next_opened_np

    assert np.allclose(np.array(gripper_target), target_np)
    assert np.allclose(np.array(gripper_velocity), velocity_np)
    assert np.array_equal(np.array(next_grasped), next_grasped_np)
    assert np.array_equal(np.array(next_opened), next_opened_np)
    assert np.allclose(np.array(next_drawer_open_amount), next_drawer_np)
    assert np.allclose(np.array(next_handle_pos), next_handle_np)


def test_locomotion_root_step_hotpath_matches_reference_math():
    dt = 0.02
    root_pos_w = mx.array([[0.1, 0.2, 0.55], [1.0, -0.2, 0.92]], dtype=mx.float32)
    root_quat_w = mx.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=mx.float32)
    root_lin_vel_b = mx.array([[0.2, -0.1, 0.05], [0.0, 0.1, -0.03]], dtype=mx.float32)
    root_ang_vel_b = mx.array([[0.01, -0.02, 0.03], [-0.02, 0.01, -0.04]], dtype=mx.float32)
    projected_gravity_b = mx.array([[0.02, -0.03, -1.0], [0.05, 0.01, -0.99]], dtype=mx.float32)
    commands = mx.array([[0.4, 0.1, 0.2], [0.1, -0.2, -0.1]], dtype=mx.float32)
    lin_gain = mx.array([0.7, 0.5], dtype=mx.float32)
    angular_acc_b = mx.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=mx.float32)
    target_height = mx.array([0.6, 1.0], dtype=mx.float32)

    outputs = locomotion_root_step_hotpath(
        dt,
        root_pos_w,
        root_quat_w,
        root_lin_vel_b,
        root_ang_vel_b,
        projected_gravity_b,
        commands,
        lin_gain,
        angular_acc_b,
        target_height,
        0.4,
        0.3,
        8.0,
        1.5,
        0.2,
    )
    reference = _locomotion_root_step_impl(
        dt,
        root_pos_w,
        root_quat_w,
        root_lin_vel_b,
        root_ang_vel_b,
        projected_gravity_b,
        commands,
        lin_gain,
        angular_acc_b,
        target_height,
        0.4,
        0.3,
        8.0,
        1.5,
        0.2,
    )
    mx.eval(*outputs, *reference)
    next_root_pos_w, next_root_quat_w, next_root_lin_vel_b, next_root_ang_vel_b, next_projected_gravity_b = outputs

    expected_lin_xy = np.array(root_lin_vel_b)[:, :2] + dt * (
        np.array(lin_gain)[:, None] * (np.array(commands)[:, :2] - np.array(root_lin_vel_b)[:, :2])
        - 0.4 * np.array(root_lin_vel_b)[:, :2]
    )
    expected_orientation_penalty = np.sum(np.square(np.array(projected_gravity_b)[:, :2]), axis=1)
    expected_z = np.array(root_lin_vel_b)[:, 2] + dt * (
        8.0 * (np.array(target_height) - np.array(root_pos_w)[:, 2])
        - 1.5 * np.array(root_lin_vel_b)[:, 2]
        - 0.2 * expected_orientation_penalty
    )
    expected_ang = np.array(root_ang_vel_b) + dt * (np.array(angular_acc_b) - 0.3 * np.array(root_ang_vel_b))

    assert np.allclose(np.array(next_root_lin_vel_b)[:, :2], expected_lin_xy)
    assert np.allclose(np.array(next_root_lin_vel_b)[:, 2], expected_z)
    assert np.allclose(np.array(next_root_ang_vel_b), expected_ang)
    for actual, expected in zip(outputs, reference, strict=True):
        assert np.allclose(np.array(actual), np.array(expected), atol=1e-6)
    assert next_root_pos_w.shape == root_pos_w.shape
    assert next_root_quat_w.shape == root_quat_w.shape
    assert next_projected_gravity_b.shape == projected_gravity_b.shape


def test_prime_contact_state_returns_zero_velocity_and_updates_contact_buffers():
    terrain = MacPlaneTerrain(num_envs=2, env_spacing=4.0, tile_size=(4.0, 4.0))
    contacts = BatchedContactSensorState(
        num_envs=2,
        body_names=("base", "LF_FOOT", "RF_FOOT"),
        foot_body_names=("LF_FOOT", "RF_FOOT"),
        step_dt=0.05,
    )
    body_pos_w = mx.array(
        [
            [[0.0, 0.0, 0.3], [0.2, 0.2, 0.005], [-0.2, 0.2, 0.006]],
            [[4.0, 0.0, 0.3], [4.2, 0.2, 0.07], [3.8, 0.2, 0.004]],
        ],
        dtype=mx.float32,
    )

    zero_body_vel = prime_contact_state(body_pos_w, contacts, terrain)
    mx.eval(zero_body_vel, contacts.contact_mask)

    assert np.allclose(np.array(zero_body_vel), 0.0)
    assert bool(contacts.contact_mask[0, 1].item()) is True
    assert bool(contacts.contact_mask[1, 2].item()) is True
