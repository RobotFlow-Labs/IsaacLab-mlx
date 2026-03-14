# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled MLX hot-path helpers for mac-native locomotion tasks."""

from __future__ import annotations

import mlx.core as mx

from .quadcopter import _quat_conjugate, _quat_from_angular_velocity, _quat_multiply, _quat_normalize, _quat_rotate


HOTPATH_BACKEND = "mlx-compiled"


def _contact_update_with_feet_impl(
    body_pos_w: mx.array,
    body_vel_w: mx.array,
    terrain_heights: mx.array,
    previous_contact: mx.array,
    previous_history: mx.array,
    previous_last_air_time: mx.array,
    previous_air_time: mx.array,
    foot_body_ids: mx.array,
    contact_margin: float,
    spring_stiffness: float,
    damping: float,
    force_threshold: float,
    step_dt: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    clearance = body_pos_w[:, :, 2] - terrain_heights
    penetration = mx.maximum(contact_margin - clearance, 0.0)
    closing_speed = mx.maximum(-body_vel_w[:, :, 2], 0.0)
    normal_force = mx.where(
        penetration > 0.0,
        spring_stiffness * penetration + damping * closing_speed,
        0.0,
    )
    zeros = mx.zeros_like(normal_force)
    forces_w = mx.stack([zeros, zeros, normal_force], axis=-1)
    current_contact = normal_force > force_threshold
    first_contact = current_contact & ~previous_contact
    history = mx.concatenate([forces_w[:, None, :, :], previous_history[:, :-1, :, :]], axis=1)

    foot_contact = current_contact[:, foot_body_ids]
    foot_first_contact = first_contact[:, foot_body_ids]
    last_air_time = mx.where(foot_first_contact, previous_air_time, previous_last_air_time)
    air_time = mx.where(foot_contact, 0.0, previous_air_time + step_dt)
    return current_contact, first_contact, history, last_air_time, air_time


def _contact_update_without_feet_impl(
    body_pos_w: mx.array,
    body_vel_w: mx.array,
    terrain_heights: mx.array,
    previous_contact: mx.array,
    previous_history: mx.array,
    contact_margin: float,
    spring_stiffness: float,
    damping: float,
    force_threshold: float,
) -> tuple[mx.array, mx.array, mx.array]:
    clearance = body_pos_w[:, :, 2] - terrain_heights
    penetration = mx.maximum(contact_margin - clearance, 0.0)
    closing_speed = mx.maximum(-body_vel_w[:, :, 2], 0.0)
    normal_force = mx.where(
        penetration > 0.0,
        spring_stiffness * penetration + damping * closing_speed,
        0.0,
    )
    zeros = mx.zeros_like(normal_force)
    forces_w = mx.stack([zeros, zeros, normal_force], axis=-1)
    current_contact = normal_force > force_threshold
    first_contact = current_contact & ~previous_contact
    history = mx.concatenate([forces_w[:, None, :, :], previous_history[:, :-1, :, :]], axis=1)
    return current_contact, first_contact, history


_contact_update_with_feet = mx.compile(_contact_update_with_feet_impl)
_contact_update_without_feet = mx.compile(_contact_update_without_feet_impl)


def contact_update_hotpath(
    body_pos_w: mx.array,
    body_vel_w: mx.array,
    terrain_heights: mx.array,
    previous_contact: mx.array,
    previous_history: mx.array,
    previous_last_air_time: mx.array,
    previous_air_time: mx.array,
    foot_body_ids: mx.array,
    *,
    contact_margin: float,
    spring_stiffness: float,
    damping: float,
    force_threshold: float,
    step_dt: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    if int(foot_body_ids.shape[0]) == 0:
        current_contact, first_contact, history = _contact_update_without_feet(
            body_pos_w,
            body_vel_w,
            terrain_heights,
            previous_contact,
            previous_history,
            contact_margin,
            spring_stiffness,
            damping,
            force_threshold,
        )
        return current_contact, first_contact, history, previous_last_air_time, previous_air_time
    return _contact_update_with_feet(
        body_pos_w,
        body_vel_w,
        terrain_heights,
        previous_contact,
        previous_history,
        previous_last_air_time,
        previous_air_time,
        foot_body_ids,
        contact_margin,
        spring_stiffness,
        damping,
        force_threshold,
        step_dt,
    )


def _quadruped_support_metrics_impl(contact_mask: mx.array, action_targets: mx.array) -> tuple[mx.array, ...]:
    support = contact_mask.astype(mx.float32)
    support_ratio = mx.mean(support, axis=1)
    left_support = mx.mean(support[:, [0, 2]], axis=1)
    right_support = mx.mean(support[:, [1, 3]], axis=1)
    front_support = mx.mean(support[:, [0, 1]], axis=1)
    rear_support = mx.mean(support[:, [2, 3]], axis=1)

    grouped_actions = action_targets.reshape((action_targets.shape[0], 4, 3))
    left_actions = mx.mean(grouped_actions[:, [0, 2], :], axis=(1, 2))
    right_actions = mx.mean(grouped_actions[:, [1, 3], :], axis=(1, 2))
    front_actions = mx.mean(grouped_actions[:, [0, 1], :], axis=(1, 2))
    rear_actions = mx.mean(grouped_actions[:, [2, 3], :], axis=(1, 2))
    return (
        support_ratio,
        left_support,
        right_support,
        front_support,
        rear_support,
        left_actions,
        right_actions,
        front_actions,
        rear_actions,
    )


def _biped_support_metrics_impl(contact_mask: mx.array, action_targets: mx.array) -> tuple[mx.array, ...]:
    support = contact_mask.astype(mx.float32)
    support_ratio = mx.mean(support, axis=1)
    left_support = support[:, 0]
    right_support = support[:, 1]
    left_actions = mx.mean(action_targets[:, :5], axis=1)
    right_actions = mx.mean(action_targets[:, 5:10], axis=1)
    return support_ratio, left_support, right_support, left_actions, right_actions


quadruped_support_metrics_hotpath = mx.compile(_quadruped_support_metrics_impl)
biped_support_metrics_hotpath = mx.compile(_biped_support_metrics_impl)


def _anymal_leg_extension_impl(joint_pos: mx.array) -> mx.array:
    num_envs = joint_pos.shape[0]
    grouped = joint_pos.reshape((num_envs, 4, 3))
    hip_pitch = grouped[:, :, 1]
    knee = grouped[:, :, 2]
    extension = 0.20 + 0.16 * mx.cos(hip_pitch) + 0.18 * mx.cos(hip_pitch + knee)
    return mx.clip(extension, 0.22, 0.62)


def _anymal_body_positions_impl(
    root_pos_w: mx.array,
    joint_pos: mx.array,
    commands: mx.array,
    gait_phase: mx.array,
    hip_offsets: mx.array,
    gait_phase_offsets: mx.array,
    foot_clearance: float,
) -> mx.array:
    num_envs = joint_pos.shape[0]
    grouped = joint_pos.reshape((num_envs, 4, 3))
    hip_abduction = grouped[:, :, 0]
    hip_pitch = grouped[:, :, 1]
    knee = grouped[:, :, 2]
    extension = _anymal_leg_extension_impl(joint_pos)
    command_speed = mx.linalg.norm(commands[:, :2], axis=1, keepdims=True)
    phase = gait_phase.reshape((num_envs, 1)) + gait_phase_offsets.reshape((1, 4))
    swing = mx.maximum(mx.sin(phase), 0.0) * (0.25 + command_speed)

    root_xy = root_pos_w[:, None, :2]
    root_z = root_pos_w[:, 2:3]
    step_x = 0.10 * commands[:, 0:1] * mx.cos(phase)
    step_y = 0.06 * commands[:, 1:2] * mx.sin(phase)
    foot_xy = root_xy + hip_offsets.reshape((1, 4, 2))
    foot_xy_x = foot_xy[:, :, 0] + step_x - 0.04 * mx.sin(hip_pitch)
    foot_xy_y = foot_xy[:, :, 1] + step_y + 0.04 * hip_abduction
    foot_xy = mx.stack([foot_xy_x, foot_xy_y], axis=-1)
    foot_z = root_z - extension + foot_clearance * swing - 0.02 * mx.tanh(knee)
    foot_pos = mx.concatenate([foot_xy, foot_z[:, :, None]], axis=-1)

    thigh_xy = root_xy + 0.55 * hip_offsets.reshape((1, 4, 2))
    thigh_xy_y = thigh_xy[:, :, 1] + 0.02 * hip_abduction
    thigh_xy = mx.stack([thigh_xy[:, :, 0], thigh_xy_y], axis=-1)
    thigh_z = root_z - 0.24 - 0.05 * mx.tanh(hip_pitch)
    thigh_pos = mx.concatenate([thigh_xy, thigh_z[:, :, None]], axis=-1)

    base_pos = root_pos_w[:, None, :]
    return mx.concatenate([base_pos, foot_pos, thigh_pos], axis=1)


def _h1_leg_extension_impl(joint_pos: mx.array) -> mx.array:
    num_envs = joint_pos.shape[0]
    leg_pos = joint_pos[:, :10].reshape((num_envs, 2, 5))
    hip_pitch = leg_pos[:, :, 2]
    knee = leg_pos[:, :, 3]
    ankle = leg_pos[:, :, 4]
    extension = (
        0.40
        + 0.20 * mx.cos(hip_pitch + 0.20)
        + 0.26 * mx.cos(hip_pitch + knee - 0.10)
        + 0.08 * mx.cos(hip_pitch + knee + ankle)
    )
    return mx.clip(extension, 0.58, 0.98)


def _h1_body_positions_impl(
    root_pos_w: mx.array,
    joint_pos: mx.array,
    commands: mx.array,
    gait_phase: mx.array,
    hip_offsets: mx.array,
    gait_phase_offsets: mx.array,
    foot_clearance: float,
) -> mx.array:
    num_envs = joint_pos.shape[0]
    leg_pos = joint_pos[:, :10].reshape((num_envs, 2, 5))
    hip_yaw = leg_pos[:, :, 0]
    hip_roll = leg_pos[:, :, 1]
    hip_pitch = leg_pos[:, :, 2]
    knee = leg_pos[:, :, 3]
    ankle = leg_pos[:, :, 4]
    extension = _h1_leg_extension_impl(joint_pos)
    command_speed = mx.linalg.norm(commands[:, :2], axis=1, keepdims=True)
    phase = gait_phase.reshape((num_envs, 1)) + gait_phase_offsets.reshape((1, 2))
    swing = mx.maximum(mx.sin(phase), 0.0) * (0.18 + 0.65 * command_speed)

    root_xy = root_pos_w[:, None, :2]
    root_z = root_pos_w[:, 2:3]
    step_x = 0.18 * commands[:, 0:1] * mx.cos(phase)
    step_y = 0.06 * commands[:, 1:2] * mx.sin(phase)

    foot_xy = root_xy + hip_offsets.reshape((1, 2, 2))
    foot_xy_x = foot_xy[:, :, 0] + step_x - 0.04 * mx.sin(hip_pitch)
    foot_xy_y = foot_xy[:, :, 1] + step_y + 0.03 * hip_roll + 0.02 * hip_yaw
    foot_xy = mx.stack([foot_xy_x, foot_xy_y], axis=-1)
    foot_z = root_z - extension + foot_clearance * swing - 0.03 * mx.tanh(ankle)
    foot_pos = mx.concatenate([foot_xy, foot_z[:, :, None]], axis=-1)

    knee_xy = root_xy + 0.5 * hip_offsets.reshape((1, 2, 2))
    knee_xy_x = knee_xy[:, :, 0] + 0.5 * step_x - 0.02 * mx.sin(hip_pitch)
    knee_xy_y = knee_xy[:, :, 1] + 0.015 * hip_roll
    knee_xy = mx.stack([knee_xy_x, knee_xy_y], axis=-1)
    knee_z = root_z - 0.42 - 0.18 * mx.tanh(hip_pitch) - 0.08 * mx.tanh(knee)
    knee_pos = mx.concatenate([knee_xy, knee_z[:, :, None]], axis=-1)

    torso_pos = root_pos_w[:, None, :]
    return mx.concatenate([torso_pos, foot_pos, knee_pos], axis=1)


anymal_leg_extension_hotpath = mx.compile(_anymal_leg_extension_impl)
anymal_body_positions_hotpath = mx.compile(_anymal_body_positions_impl)
h1_leg_extension_hotpath = mx.compile(_h1_leg_extension_impl)
h1_body_positions_hotpath = mx.compile(_h1_body_positions_impl)


def _franka_end_effector_position_impl(joint_pos: mx.array) -> mx.array:
    q0 = joint_pos[:, 0]
    q1 = joint_pos[:, 1]
    q2 = joint_pos[:, 2]
    q3 = joint_pos[:, 3]
    q4 = joint_pos[:, 4]
    q5 = joint_pos[:, 5]
    q6 = joint_pos[:, 6]
    x = (
        0.28
        + 0.18 * mx.cos(q0) * mx.cos(q1)
        + 0.14 * mx.cos(q0) * mx.cos(q1 + q2)
        + 0.08 * mx.cos(q0) * mx.cos(q1 + q2 + 0.5 * q3)
        - 0.02 * mx.sin(q4)
    )
    y = 0.20 * mx.sin(q0) + 0.07 * mx.sin(q0 + 0.5 * q3) + 0.03 * mx.tanh(q5) + 0.015 * mx.sin(q6)
    z = 0.28 + 0.18 * mx.sin(-q1) + 0.13 * mx.sin(-(q1 + q2)) + 0.07 * mx.sin(-(q1 + q2 + 0.5 * q3))
    return mx.stack((x, y, z), axis=-1).astype(mx.float32)


def _franka_lift_object_step_impl(
    cube_pos_w: mx.array,
    ee_pos_w: mx.array,
    gripper_joint_pos: mx.array,
    gripper_action: mx.array,
    grasped: mx.array,
    physics_dt: float,
    gripper_lower_limit: float,
    gripper_upper_limit: float,
    gripper_closed_threshold: float,
    grasp_distance_threshold: float,
    grasp_offset_z: float,
    table_height: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    gripper_target = mx.where(gripper_action < 0.0, gripper_lower_limit, gripper_upper_limit).astype(mx.float32)
    gripper_velocity = (gripper_target - gripper_joint_pos) / physics_dt
    dist_to_cube = mx.linalg.norm(cube_pos_w - ee_pos_w, axis=1)
    gripper_closed = gripper_target <= gripper_closed_threshold
    can_grasp = dist_to_cube <= grasp_distance_threshold
    next_grasped = (grasped | (can_grasp & gripper_closed)) & gripper_closed
    attached_cube = ee_pos_w + mx.array([0.0, 0.0, -grasp_offset_z], dtype=mx.float32)
    resting_cube = mx.stack(
        (
            cube_pos_w[:, 0],
            cube_pos_w[:, 1],
            mx.maximum(table_height, cube_pos_w[:, 2] - physics_dt * 0.35),
        ),
        axis=-1,
    )
    next_cube_pos = mx.where(next_grasped[:, None], attached_cube, resting_cube)
    return gripper_target, gripper_velocity.astype(mx.float32), next_grasped.astype(mx.bool_), next_cube_pos.astype(mx.float32)


def _franka_stack_object_step_impl(
    cube_pos_w: mx.array,
    support_cube_pos_w: mx.array,
    ee_pos_w: mx.array,
    gripper_joint_pos: mx.array,
    gripper_action: mx.array,
    grasped: mx.array,
    stacked: mx.array,
    physics_dt: float,
    gripper_lower_limit: float,
    gripper_upper_limit: float,
    gripper_closed_threshold: float,
    stack_release_open_threshold: float,
    grasp_distance_threshold: float,
    grasp_offset_z: float,
    table_height: float,
    stack_offset_z: float,
    stack_xy_threshold: float,
    stack_z_threshold: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    gripper_target = mx.where(gripper_action < 0.0, gripper_lower_limit, gripper_upper_limit).astype(mx.float32)
    gripper_velocity = (gripper_target - gripper_joint_pos) / physics_dt
    dist_to_cube = mx.linalg.norm(cube_pos_w - ee_pos_w, axis=1)
    gripper_closed = gripper_target <= gripper_closed_threshold
    gripper_open = gripper_target >= stack_release_open_threshold
    can_grasp = dist_to_cube <= grasp_distance_threshold
    next_grasped = (grasped | (can_grasp & gripper_closed)) & gripper_closed & ~stacked

    attached_cube = ee_pos_w + mx.array([0.0, 0.0, -grasp_offset_z], dtype=mx.float32)
    stack_target = support_cube_pos_w + mx.array([0.0, 0.0, stack_offset_z], dtype=mx.float32)
    release_cube = mx.where(grasped[:, None], attached_cube, cube_pos_w)
    stack_xy_error = mx.linalg.norm(release_cube[:, :2] - stack_target[:, :2], axis=1)
    stack_z_error = mx.abs(release_cube[:, 2] - stack_target[:, 2])
    newly_stacked = grasped & gripper_open & (stack_xy_error <= stack_xy_threshold) & (stack_z_error <= stack_z_threshold)
    next_stacked = stacked | newly_stacked

    resting_cube = mx.stack(
        (
            release_cube[:, 0],
            release_cube[:, 1],
            mx.maximum(table_height, release_cube[:, 2] - physics_dt * 0.35),
        ),
        axis=-1,
    )
    next_cube_pos = mx.where(
        next_stacked[:, None],
        stack_target,
        mx.where(next_grasped[:, None], attached_cube, resting_cube),
    )
    next_grasped = next_grasped & ~next_stacked
    return (
        gripper_target,
        gripper_velocity.astype(mx.float32),
        next_grasped.astype(mx.bool_),
        next_stacked.astype(mx.bool_),
        next_cube_pos.astype(mx.float32),
    )


def _franka_cabinet_step_impl(
    handle_anchor_pos_w: mx.array,
    ee_pos_w: mx.array,
    gripper_joint_pos: mx.array,
    gripper_action: mx.array,
    grasped: mx.array,
    opened: mx.array,
    drawer_open_amount: mx.array,
    physics_dt: float,
    gripper_lower_limit: float,
    gripper_upper_limit: float,
    gripper_closed_threshold: float,
    handle_grasp_threshold: float,
    drawer_open_distance_max: float,
    drawer_success_distance: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    gripper_target = mx.where(gripper_action < 0.0, gripper_lower_limit, gripper_upper_limit).astype(mx.float32)
    gripper_velocity = (gripper_target - gripper_joint_pos) / physics_dt
    current_handle_pos = handle_anchor_pos_w + mx.stack(
        (
            drawer_open_amount,
            mx.zeros_like(drawer_open_amount),
            mx.zeros_like(drawer_open_amount),
        ),
        axis=-1,
    )
    dist_to_handle = mx.linalg.norm(current_handle_pos - ee_pos_w, axis=1)
    gripper_closed = gripper_target <= gripper_closed_threshold
    can_grasp = dist_to_handle <= handle_grasp_threshold
    next_grasped = (grasped | (can_grasp & gripper_closed)) & gripper_closed & ~opened
    pulled_drawer_open_amount = mx.clip(
        ee_pos_w[:, 0] - handle_anchor_pos_w[:, 0],
        0.0,
        drawer_open_distance_max,
    )
    next_drawer_open_amount = mx.where(next_grasped, pulled_drawer_open_amount, drawer_open_amount).astype(mx.float32)
    next_opened = opened | (next_drawer_open_amount >= drawer_success_distance)
    next_handle_pos = handle_anchor_pos_w + mx.stack(
        (
            next_drawer_open_amount,
            mx.zeros_like(next_drawer_open_amount),
            mx.zeros_like(next_drawer_open_amount),
        ),
        axis=-1,
    )
    next_grasped = next_grasped & ~next_opened
    return (
        gripper_target,
        gripper_velocity.astype(mx.float32),
        next_grasped.astype(mx.bool_),
        next_opened.astype(mx.bool_),
        next_drawer_open_amount,
        next_handle_pos.astype(mx.float32),
    )


def _franka_stack_rgb_step_impl(
    middle_cube_pos_w: mx.array,
    top_cube_pos_w: mx.array,
    support_cube_pos_w: mx.array,
    ee_pos_w: mx.array,
    gripper_joint_pos: mx.array,
    gripper_action: mx.array,
    middle_grasped: mx.array,
    top_grasped: mx.array,
    middle_stacked: mx.array,
    top_stacked: mx.array,
    physics_dt: float,
    gripper_lower_limit: float,
    gripper_upper_limit: float,
    gripper_closed_threshold: float,
    stack_release_open_threshold: float,
    grasp_distance_threshold: float,
    grasp_offset_z: float,
    table_height: float,
    stack_offset_z: float,
    stack_xy_threshold: float,
    stack_z_threshold: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    gripper_target = mx.where(gripper_action < 0.0, gripper_lower_limit, gripper_upper_limit).astype(mx.float32)
    gripper_velocity = (gripper_target - gripper_joint_pos) / physics_dt
    use_top_cube = middle_stacked

    active_cube_pos = mx.where(use_top_cube[:, None], top_cube_pos_w, middle_cube_pos_w)
    active_target = mx.where(
        use_top_cube[:, None],
        middle_cube_pos_w + mx.array([0.0, 0.0, stack_offset_z], dtype=mx.float32),
        support_cube_pos_w + mx.array([0.0, 0.0, stack_offset_z], dtype=mx.float32),
    )
    active_grasped = mx.where(use_top_cube, top_grasped, middle_grasped)

    dist_to_cube = mx.linalg.norm(active_cube_pos - ee_pos_w, axis=1)
    gripper_closed = gripper_target <= gripper_closed_threshold
    gripper_open = gripper_target >= stack_release_open_threshold
    can_grasp = dist_to_cube <= grasp_distance_threshold
    next_active_grasped = (active_grasped | (can_grasp & gripper_closed)) & gripper_closed & ~top_stacked

    attached_cube = ee_pos_w + mx.array([0.0, 0.0, -grasp_offset_z], dtype=mx.float32)
    release_cube = mx.where(active_grasped[:, None], attached_cube, active_cube_pos)
    stack_xy_error = mx.linalg.norm(release_cube[:, :2] - active_target[:, :2], axis=1)
    stack_z_error = mx.abs(release_cube[:, 2] - active_target[:, 2])
    newly_stacked = active_grasped & gripper_open & (stack_xy_error <= stack_xy_threshold) & (stack_z_error <= stack_z_threshold)

    next_middle_stacked = middle_stacked | (~use_top_cube & newly_stacked)
    next_top_stacked = top_stacked | (use_top_cube & newly_stacked)

    resting_cube = mx.stack(
        (
            release_cube[:, 0],
            release_cube[:, 1],
            mx.maximum(table_height, release_cube[:, 2] - physics_dt * 0.35),
        ),
        axis=-1,
    )
    next_active_cube_pos = mx.where(
        newly_stacked[:, None],
        active_target,
        mx.where(next_active_grasped[:, None], attached_cube, resting_cube),
    )

    next_middle_cube_pos = mx.where(use_top_cube[:, None], middle_cube_pos_w, next_active_cube_pos).astype(mx.float32)
    next_top_cube_pos = mx.where(use_top_cube[:, None], next_active_cube_pos, top_cube_pos_w).astype(mx.float32)

    next_middle_grasped = mx.where(use_top_cube, middle_grasped, next_active_grasped & ~next_middle_stacked).astype(mx.bool_)
    next_top_grasped = mx.where(use_top_cube, next_active_grasped & ~next_top_stacked, top_grasped).astype(mx.bool_)
    return (
        gripper_target,
        gripper_velocity.astype(mx.float32),
        next_middle_grasped,
        next_top_grasped,
        next_middle_stacked.astype(mx.bool_),
        next_top_stacked.astype(mx.bool_),
        next_middle_cube_pos,
        next_top_cube_pos,
    )


_franka_end_effector_position_compiled = mx.compile(_franka_end_effector_position_impl)
_franka_end_effector_position_metal = None
_franka_end_effector_hotpath_initialized = False
FRANKA_HOTPATH_BACKEND = HOTPATH_BACKEND
_locomotion_root_step_metal = None
_locomotion_root_step_hotpath_initialized = False
LOCOMOTION_HOTPATH_BACKEND = HOTPATH_BACKEND


def _build_franka_end_effector_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float q0 = inp[env_id * 7 + 0];
        float q1 = inp[env_id * 7 + 1];
        float q2 = inp[env_id * 7 + 2];
        float q3 = inp[env_id * 7 + 3];
        float q4 = inp[env_id * 7 + 4];
        float q5 = inp[env_id * 7 + 5];
        float q6 = inp[env_id * 7 + 6];
        out[env_id * 3 + 0] = 0.28f
            + 0.18f * metal::cos(q0) * metal::cos(q1)
            + 0.14f * metal::cos(q0) * metal::cos(q1 + q2)
            + 0.08f * metal::cos(q0) * metal::cos(q1 + q2 + 0.5f * q3)
            - 0.02f * metal::sin(q4);
        out[env_id * 3 + 1] = 0.20f * metal::sin(q0)
            + 0.07f * metal::sin(q0 + 0.5f * q3)
            + 0.03f * metal::tanh(q5)
            + 0.015f * metal::sin(q6);
        out[env_id * 3 + 2] = 0.28f
            + 0.18f * metal::sin(-q1)
            + 0.13f * metal::sin(-(q1 + q2))
            + 0.07f * metal::sin(-(q1 + q2 + 0.5f * q3));
    """
    try:
        return mx.fast.metal_kernel(
            name="franka_end_effector_position",
            input_names=["inp"],
            output_names=["out"],
            source=source,
        )
    except Exception:
        return None

def _ensure_franka_end_effector_hotpath() -> None:
    global _franka_end_effector_position_metal
    global _franka_end_effector_hotpath_initialized
    global FRANKA_HOTPATH_BACKEND
    if _franka_end_effector_hotpath_initialized:
        return
    _franka_end_effector_hotpath_initialized = True
    _franka_end_effector_position_metal = _build_franka_end_effector_metal_kernel()
    if _franka_end_effector_position_metal is not None:
        FRANKA_HOTPATH_BACKEND = "mlx-metal-ee"


def get_franka_hotpath_backend() -> str:
    _ensure_franka_end_effector_hotpath()
    return FRANKA_HOTPATH_BACKEND


def franka_end_effector_position_hotpath(joint_pos: mx.array) -> mx.array:
    _ensure_franka_end_effector_hotpath()
    joint_pos = mx.array(joint_pos, dtype=mx.float32)
    if _franka_end_effector_position_metal is None:
        return _franka_end_effector_position_compiled(joint_pos)
    if int(joint_pos.shape[0]) == 0:
        return mx.zeros((0, 3), dtype=mx.float32)
    return _franka_end_effector_position_metal(
        inputs=[joint_pos],
        grid=(int(joint_pos.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(int(joint_pos.shape[0]), 3)],
        output_dtypes=[mx.float32],
    )[0]


franka_lift_object_step_hotpath = mx.compile(_franka_lift_object_step_impl)
franka_stack_object_step_hotpath = mx.compile(_franka_stack_object_step_impl)
franka_cabinet_step_hotpath = mx.compile(_franka_cabinet_step_impl)
franka_stack_rgb_step_hotpath = mx.compile(_franka_stack_rgb_step_impl)


def _locomotion_root_step_impl(
    dt: float,
    root_pos_w: mx.array,
    root_quat_w: mx.array,
    root_lin_vel_b: mx.array,
    root_ang_vel_b: mx.array,
    projected_gravity_b: mx.array,
    commands: mx.array,
    lin_gain: mx.array,
    angular_acc_b: mx.array,
    target_height: mx.array,
    root_lin_damping: float,
    root_ang_damping: float,
    height_stiffness: float,
    height_damping: float,
    orientation_height_penalty: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    lin_xy = root_lin_vel_b[:, :2]
    next_lin_xy = lin_xy + dt * (lin_gain[:, None] * (commands[:, :2] - lin_xy) - root_lin_damping * lin_xy)
    orientation_penalty = mx.sum(mx.square(projected_gravity_b[:, :2]), axis=1)
    z_acc = (
        height_stiffness * (target_height - root_pos_w[:, 2])
        - height_damping * root_lin_vel_b[:, 2]
        - orientation_height_penalty * orientation_penalty
    )
    next_lin_vel_b = mx.concatenate([next_lin_xy, (root_lin_vel_b[:, 2] + dt * z_acc)[:, None]], axis=1)

    next_ang_vel_b = root_ang_vel_b + dt * (angular_acc_b - root_ang_damping * root_ang_vel_b)
    delta_quat = _quat_from_angular_velocity(next_ang_vel_b, dt)
    next_root_quat_w = _quat_normalize(_quat_multiply(root_quat_w, delta_quat))
    world_vel = _quat_rotate(next_root_quat_w, next_lin_vel_b)
    next_root_pos_w = root_pos_w + dt * world_vel

    gravity_w = mx.broadcast_to(mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32), next_lin_vel_b.shape)
    next_projected_gravity_b = _quat_rotate(_quat_conjugate(next_root_quat_w), gravity_w)
    return next_root_pos_w, next_root_quat_w, next_lin_vel_b, next_ang_vel_b, next_projected_gravity_b


_locomotion_root_step_compiled = mx.compile(_locomotion_root_step_impl)


def _build_locomotion_root_step_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float dt = params[0];
        float root_lin_damping = params[1];
        float root_ang_damping = params[2];
        float height_stiffness = params[3];
        float height_damping = params[4];
        float orientation_height_penalty = params[5];

        float3 root_pos = float3(root_pos_w[env_id * 3 + 0], root_pos_w[env_id * 3 + 1], root_pos_w[env_id * 3 + 2]);
        float4 root_quat = float4(
            root_quat_w[env_id * 4 + 0],
            root_quat_w[env_id * 4 + 1],
            root_quat_w[env_id * 4 + 2],
            root_quat_w[env_id * 4 + 3]
        );
        float3 root_lin = float3(
            root_lin_vel_b[env_id * 3 + 0],
            root_lin_vel_b[env_id * 3 + 1],
            root_lin_vel_b[env_id * 3 + 2]
        );
        float3 root_ang = float3(
            root_ang_vel_b[env_id * 3 + 0],
            root_ang_vel_b[env_id * 3 + 1],
            root_ang_vel_b[env_id * 3 + 2]
        );
        float3 projected_gravity = float3(
            projected_gravity_b[env_id * 3 + 0],
            projected_gravity_b[env_id * 3 + 1],
            projected_gravity_b[env_id * 3 + 2]
        );
        float3 command = float3(commands[env_id * 3 + 0], commands[env_id * 3 + 1], commands[env_id * 3 + 2]);
        float lin_gain_value = lin_gain[env_id];
        float3 angular_acc = float3(
            angular_acc_b[env_id * 3 + 0],
            angular_acc_b[env_id * 3 + 1],
            angular_acc_b[env_id * 3 + 2]
        );
        float target_height_value = target_height[env_id];

        float2 lin_xy = float2(root_lin.x, root_lin.y);
        float2 next_lin_xy = lin_xy + dt * (
            lin_gain_value * (float2(command.x, command.y) - lin_xy) - root_lin_damping * lin_xy
        );
        float orientation_penalty = projected_gravity.x * projected_gravity.x + projected_gravity.y * projected_gravity.y;
        float z_acc = height_stiffness * (target_height_value - root_pos.z)
            - height_damping * root_lin.z
            - orientation_height_penalty * orientation_penalty;
        float3 next_lin = float3(next_lin_xy.x, next_lin_xy.y, root_lin.z + dt * z_acc);
        float3 next_ang = root_ang + dt * (angular_acc - root_ang_damping * root_ang);

        float3 delta = next_ang * dt;
        float angle = metal::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
        float half_angle = 0.5f * angle;
        float3 axis = angle > 1e-8f ? delta / angle : float3(0.0f);
        float sin_half = metal::sin(half_angle);
        float cos_half = metal::cos(half_angle);
        float4 delta_quat = float4(axis.x * sin_half, axis.y * sin_half, axis.z * sin_half, cos_half);
        float delta_norm = metal::sqrt(
            delta_quat.x * delta_quat.x
            + delta_quat.y * delta_quat.y
            + delta_quat.z * delta_quat.z
            + delta_quat.w * delta_quat.w
        );
        delta_quat = delta_quat / metal::max(delta_norm, 1e-8f);

        float4 next_quat = float4(
            root_quat.w * delta_quat.x + root_quat.x * delta_quat.w + root_quat.y * delta_quat.z - root_quat.z * delta_quat.y,
            root_quat.w * delta_quat.y - root_quat.x * delta_quat.z + root_quat.y * delta_quat.w + root_quat.z * delta_quat.x,
            root_quat.w * delta_quat.z + root_quat.x * delta_quat.y - root_quat.y * delta_quat.x + root_quat.z * delta_quat.w,
            root_quat.w * delta_quat.w - root_quat.x * delta_quat.x - root_quat.y * delta_quat.y - root_quat.z * delta_quat.z
        );
        float next_quat_norm = metal::sqrt(
            next_quat.x * next_quat.x + next_quat.y * next_quat.y + next_quat.z * next_quat.z + next_quat.w * next_quat.w
        );
        next_quat = next_quat / metal::max(next_quat_norm, 1e-8f);

        float3 quat_xyz = float3(next_quat.x, next_quat.y, next_quat.z);
        float quat_w = next_quat.w;
        float3 cross1 = metal::cross(quat_xyz, next_lin);
        float3 cross2 = metal::cross(quat_xyz, cross1);
        float3 world_vel = next_lin + 2.0f * (quat_w * cross1 + cross2);
        float3 next_pos = root_pos + dt * world_vel;

        float3 conj_xyz = -quat_xyz;
        float3 gravity = float3(0.0f, 0.0f, -1.0f);
        float3 gcross1 = metal::cross(conj_xyz, gravity);
        float3 gcross2 = metal::cross(conj_xyz, gcross1);
        float3 next_gravity = gravity + 2.0f * (quat_w * gcross1 + gcross2);

        next_root_pos_w[env_id * 3 + 0] = next_pos.x;
        next_root_pos_w[env_id * 3 + 1] = next_pos.y;
        next_root_pos_w[env_id * 3 + 2] = next_pos.z;
        next_root_quat_w[env_id * 4 + 0] = next_quat.x;
        next_root_quat_w[env_id * 4 + 1] = next_quat.y;
        next_root_quat_w[env_id * 4 + 2] = next_quat.z;
        next_root_quat_w[env_id * 4 + 3] = next_quat.w;
        next_root_lin_vel_b[env_id * 3 + 0] = next_lin.x;
        next_root_lin_vel_b[env_id * 3 + 1] = next_lin.y;
        next_root_lin_vel_b[env_id * 3 + 2] = next_lin.z;
        next_root_ang_vel_b[env_id * 3 + 0] = next_ang.x;
        next_root_ang_vel_b[env_id * 3 + 1] = next_ang.y;
        next_root_ang_vel_b[env_id * 3 + 2] = next_ang.z;
        next_projected_gravity_b[env_id * 3 + 0] = next_gravity.x;
        next_projected_gravity_b[env_id * 3 + 1] = next_gravity.y;
        next_projected_gravity_b[env_id * 3 + 2] = next_gravity.z;
    """
    try:
        return mx.fast.metal_kernel(
            name="locomotion_root_step",
            input_names=[
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_b",
                "root_ang_vel_b",
                "projected_gravity_b",
                "commands",
                "lin_gain",
                "angular_acc_b",
                "target_height",
                "params",
            ],
            output_names=[
                "next_root_pos_w",
                "next_root_quat_w",
                "next_root_lin_vel_b",
                "next_root_ang_vel_b",
                "next_projected_gravity_b",
            ],
            source=source,
        )
    except Exception:
        return None


def _ensure_locomotion_root_hotpath() -> None:
    global _locomotion_root_step_metal
    global _locomotion_root_step_hotpath_initialized
    global LOCOMOTION_HOTPATH_BACKEND
    if _locomotion_root_step_hotpath_initialized:
        return
    _locomotion_root_step_hotpath_initialized = True
    _locomotion_root_step_metal = _build_locomotion_root_step_metal_kernel()
    if _locomotion_root_step_metal is not None:
        LOCOMOTION_HOTPATH_BACKEND = "mlx-metal-root-step"


def get_locomotion_hotpath_backend() -> str:
    _ensure_locomotion_root_hotpath()
    return LOCOMOTION_HOTPATH_BACKEND


def locomotion_root_step_hotpath(
    dt: float,
    root_pos_w: mx.array,
    root_quat_w: mx.array,
    root_lin_vel_b: mx.array,
    root_ang_vel_b: mx.array,
    projected_gravity_b: mx.array,
    commands: mx.array,
    lin_gain: mx.array,
    angular_acc_b: mx.array,
    target_height: mx.array,
    root_lin_damping: float,
    root_ang_damping: float,
    height_stiffness: float,
    height_damping: float,
    orientation_height_penalty: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    _ensure_locomotion_root_hotpath()
    root_pos_w = mx.array(root_pos_w, dtype=mx.float32)
    root_quat_w = mx.array(root_quat_w, dtype=mx.float32)
    root_lin_vel_b = mx.array(root_lin_vel_b, dtype=mx.float32)
    root_ang_vel_b = mx.array(root_ang_vel_b, dtype=mx.float32)
    projected_gravity_b = mx.array(projected_gravity_b, dtype=mx.float32)
    commands = mx.array(commands, dtype=mx.float32)
    lin_gain = mx.array(lin_gain, dtype=mx.float32)
    angular_acc_b = mx.array(angular_acc_b, dtype=mx.float32)
    target_height = mx.array(target_height, dtype=mx.float32)
    if _locomotion_root_step_metal is None:
        return _locomotion_root_step_compiled(
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
            root_lin_damping,
            root_ang_damping,
            height_stiffness,
            height_damping,
            orientation_height_penalty,
        )
    if int(root_pos_w.shape[0]) == 0:
        return (
            mx.zeros_like(root_pos_w),
            mx.zeros_like(root_quat_w),
            mx.zeros_like(root_lin_vel_b),
            mx.zeros_like(root_ang_vel_b),
            mx.zeros_like(projected_gravity_b),
        )
    params = mx.array(
        [dt, root_lin_damping, root_ang_damping, height_stiffness, height_damping, orientation_height_penalty],
        dtype=mx.float32,
    )
    outputs = _locomotion_root_step_metal(
        inputs=[
            root_pos_w,
            root_quat_w,
            root_lin_vel_b,
            root_ang_vel_b,
            projected_gravity_b,
            commands,
            lin_gain,
            angular_acc_b,
            target_height,
            params,
        ],
        grid=(int(root_pos_w.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[
            tuple(root_pos_w.shape),
            tuple(root_quat_w.shape),
            tuple(root_lin_vel_b.shape),
            tuple(root_ang_vel_b.shape),
            tuple(projected_gravity_b.shape),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


def prime_contact_state(
    body_pos_w: mx.array,
    contact_model,
    terrain,
    *,
    env_ids: list[int] | None = None,
) -> mx.array:
    """Prime a contact model with zero body velocity at the provided body positions."""
    zero_body_vel = mx.zeros_like(body_pos_w)
    contact_model.update(body_pos_w, zero_body_vel, terrain, env_ids=env_ids)
    return zero_body_vel
