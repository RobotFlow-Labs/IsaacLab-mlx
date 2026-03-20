# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled MLX hot-path helpers for mac-native locomotion tasks."""

from __future__ import annotations

import math

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


_contact_update_with_feet_compiled = mx.compile(_contact_update_with_feet_impl)
_contact_update_without_feet_compiled = mx.compile(_contact_update_without_feet_impl)
CONTACT_HOTPATH_BACKEND = HOTPATH_BACKEND
_contact_update_with_feet_metal = None
_contact_update_hotpath_initialized = False


def _build_contact_update_with_feet_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float contact_margin = params[0];
        float spring_stiffness = params[1];
        float damping = params[2];
        float force_threshold = params[3];
        float step_dt = params[4];
        uint body_count = (uint)params[5];
        uint foot_count = (uint)params[6];
        uint history_length = (uint)params[7];

        uint body_base = env_id * body_count * 3;
        uint contact_base = env_id * body_count;
        uint history_base = env_id * history_length * body_count * 3;
        uint foot_base = env_id * foot_count;

        for (uint body_id = 0; body_id < body_count; ++body_id) {
            uint body_offset = body_base + body_id * 3;
            float clearance = body_pos_w[body_offset + 2] - terrain_heights[contact_base + body_id];
            float penetration = contact_margin - clearance;
            if (penetration < 0.0f) {
                penetration = 0.0f;
            }
            float closing_speed = -body_vel_w[body_offset + 2];
            if (closing_speed < 0.0f) {
                closing_speed = 0.0f;
            }
            float normal_force = penetration > 0.0f ? spring_stiffness * penetration + damping * closing_speed : 0.0f;
            float current_contact_value = normal_force > force_threshold ? 1.0f : 0.0f;
            float first_contact_value = current_contact_value > 0.5f && previous_contact[contact_base + body_id] <= 0.5f ? 1.0f : 0.0f;
            current_contact_out[contact_base + body_id] = current_contact_value;
            first_contact_out[contact_base + body_id] = first_contact_value;

            uint history_row_base = history_base + body_id * 3;
            history_out[history_row_base + 0] = 0.0f;
            history_out[history_row_base + 1] = 0.0f;
            history_out[history_row_base + 2] = normal_force;

            for (uint history_index = 1; history_index < history_length; ++history_index) {
                uint prev_history_base = history_base + (history_index - 1) * body_count * 3 + body_id * 3;
                uint next_history_base = history_base + history_index * body_count * 3 + body_id * 3;
                history_out[next_history_base + 0] = previous_history[prev_history_base + 0];
                history_out[next_history_base + 1] = previous_history[prev_history_base + 1];
                history_out[next_history_base + 2] = previous_history[prev_history_base + 2];
            }
        }

        for (uint foot_index = 0; foot_index < foot_count; ++foot_index) {
            uint body_id = (uint)foot_body_ids[foot_index];
            uint foot_offset = foot_base + foot_index;
            float current_contact_value = current_contact_out[contact_base + body_id];
            float first_contact_value = first_contact_out[contact_base + body_id];
            last_air_time_out[foot_offset] = first_contact_value > 0.5f ? previous_air_time[foot_offset] : previous_last_air_time[foot_offset];
            air_time_out[foot_offset] = current_contact_value > 0.5f ? 0.0f : previous_air_time[foot_offset] + step_dt;
        }
    """
    try:
        return mx.fast.metal_kernel(
            name="contact_update_with_feet",
            input_names=[
                "body_pos_w",
                "body_vel_w",
                "terrain_heights",
                "previous_contact",
                "previous_history",
                "previous_last_air_time",
                "previous_air_time",
                "foot_body_ids",
                "params",
            ],
            output_names=[
                "current_contact_out",
                "first_contact_out",
                "history_out",
                "last_air_time_out",
                "air_time_out",
            ],
            source=source,
        )
    except Exception:
        return None


def _ensure_contact_update_hotpath() -> None:
    global _contact_update_with_feet_metal
    global _contact_update_hotpath_initialized
    global CONTACT_HOTPATH_BACKEND
    if _contact_update_hotpath_initialized:
        return
    _contact_update_hotpath_initialized = True
    _contact_update_with_feet_metal = _build_contact_update_with_feet_metal_kernel()
    if _contact_update_with_feet_metal is not None:
        CONTACT_HOTPATH_BACKEND = "mlx-metal-contact"


def get_contact_hotpath_backend() -> str:
    _ensure_contact_update_hotpath()
    return CONTACT_HOTPATH_BACKEND


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
    _ensure_contact_update_hotpath()
    if int(foot_body_ids.shape[0]) == 0:
        current_contact, first_contact, history = _contact_update_without_feet_compiled(
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
    if _contact_update_with_feet_metal is None:
        return _contact_update_with_feet_compiled(
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
    body_pos_w = mx.array(body_pos_w, dtype=mx.float32)
    body_vel_w = mx.array(body_vel_w, dtype=mx.float32)
    terrain_heights = mx.array(terrain_heights, dtype=mx.float32)
    previous_contact = mx.array(previous_contact, dtype=mx.float32)
    previous_history = mx.array(previous_history, dtype=mx.float32)
    previous_last_air_time = mx.array(previous_last_air_time, dtype=mx.float32)
    previous_air_time = mx.array(previous_air_time, dtype=mx.float32)
    foot_body_ids = mx.array(foot_body_ids, dtype=mx.int32)
    if int(body_pos_w.shape[0]) == 0:
        return (
            mx.zeros((0, body_pos_w.shape[1]), dtype=mx.bool_),
            mx.zeros((0, body_pos_w.shape[1]), dtype=mx.bool_),
            mx.zeros((0, previous_history.shape[1], body_pos_w.shape[1], 3), dtype=mx.float32),
            mx.zeros((0, foot_body_ids.shape[0]), dtype=mx.float32),
            mx.zeros((0, foot_body_ids.shape[0]), dtype=mx.float32),
        )
    params = mx.array(
        [
            contact_margin,
            spring_stiffness,
            damping,
            force_threshold,
            step_dt,
            float(body_pos_w.shape[1]),
            float(foot_body_ids.shape[0]),
            float(previous_history.shape[1]),
        ],
        dtype=mx.float32,
    )
    outputs = _contact_update_with_feet_metal(
        inputs=[
            body_pos_w,
            body_vel_w,
            terrain_heights,
            previous_contact,
            previous_history,
            previous_last_air_time,
            previous_air_time,
            foot_body_ids,
            params,
        ],
        grid=(int(body_pos_w.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[
            (int(body_pos_w.shape[0]), int(body_pos_w.shape[1])),
            (int(body_pos_w.shape[0]), int(body_pos_w.shape[1])),
            tuple(previous_history.shape),
            (int(body_pos_w.shape[0]), int(foot_body_ids.shape[0])),
            (int(body_pos_w.shape[0]), int(foot_body_ids.shape[0])),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return (
        outputs[0].astype(mx.bool_),
        outputs[1].astype(mx.bool_),
        outputs[2],
        outputs[3],
        outputs[4],
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


_quadruped_support_metrics_compiled = mx.compile(_quadruped_support_metrics_impl)
_biped_support_metrics_compiled = mx.compile(_biped_support_metrics_impl)
_quadruped_support_metrics_metal = None
_biped_support_metrics_metal = None


def _build_quadruped_support_metrics_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        uint contact_base = env_id * 4;
        uint action_base = env_id * 12;

        float contact_0 = contact_mask[contact_base + 0] > 0.5f ? 1.0f : 0.0f;
        float contact_1 = contact_mask[contact_base + 1] > 0.5f ? 1.0f : 0.0f;
        float contact_2 = contact_mask[contact_base + 2] > 0.5f ? 1.0f : 0.0f;
        float contact_3 = contact_mask[contact_base + 3] > 0.5f ? 1.0f : 0.0f;
        float support_ratio = (contact_0 + contact_1 + contact_2 + contact_3) / 4.0f;
        float left_support = (contact_0 + contact_2) / 2.0f;
        float right_support = (contact_1 + contact_3) / 2.0f;
        float front_support = (contact_0 + contact_1) / 2.0f;
        float rear_support = (contact_2 + contact_3) / 2.0f;

        float left_actions = (action_targets[action_base + 0] + action_targets[action_base + 1] + action_targets[action_base + 2]
            + action_targets[action_base + 6] + action_targets[action_base + 7] + action_targets[action_base + 8]) / 6.0f;
        float right_actions = (action_targets[action_base + 3] + action_targets[action_base + 4] + action_targets[action_base + 5]
            + action_targets[action_base + 9] + action_targets[action_base + 10] + action_targets[action_base + 11]) / 6.0f;
        float front_actions = (action_targets[action_base + 0] + action_targets[action_base + 1] + action_targets[action_base + 2]
            + action_targets[action_base + 3] + action_targets[action_base + 4] + action_targets[action_base + 5]) / 6.0f;
        float rear_actions = (action_targets[action_base + 6] + action_targets[action_base + 7] + action_targets[action_base + 8]
            + action_targets[action_base + 9] + action_targets[action_base + 10] + action_targets[action_base + 11]) / 6.0f;

        support_ratio_out[env_id] = support_ratio;
        left_support_out[env_id] = left_support;
        right_support_out[env_id] = right_support;
        front_support_out[env_id] = front_support;
        rear_support_out[env_id] = rear_support;
        left_actions_out[env_id] = left_actions;
        right_actions_out[env_id] = right_actions;
        front_actions_out[env_id] = front_actions;
        rear_actions_out[env_id] = rear_actions;
    """
    try:
        return mx.fast.metal_kernel(
            name="quadruped_support_metrics",
            input_names=["contact_mask", "action_targets"],
            output_names=[
                "support_ratio_out",
                "left_support_out",
                "right_support_out",
                "front_support_out",
                "rear_support_out",
                "left_actions_out",
                "right_actions_out",
                "front_actions_out",
                "rear_actions_out",
            ],
            source=source,
        )
    except Exception:
        return None


def _build_biped_support_metrics_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        uint contact_base = env_id * 2;
        uint action_base = env_id * 19;

        float contact_0 = contact_mask[contact_base + 0] > 0.5f ? 1.0f : 0.0f;
        float contact_1 = contact_mask[contact_base + 1] > 0.5f ? 1.0f : 0.0f;
        float support_ratio = (contact_0 + contact_1) / 2.0f;
        float left_support = contact_0;
        float right_support = contact_1;

        float left_actions = 0.0f;
        float right_actions = 0.0f;
        for (uint i = 0; i < 5; ++i) {
            left_actions += action_targets[action_base + i];
            right_actions += action_targets[action_base + 5 + i];
        }
        left_actions /= 5.0f;
        right_actions /= 5.0f;

        support_ratio_out[env_id] = support_ratio;
        left_support_out[env_id] = left_support;
        right_support_out[env_id] = right_support;
        left_actions_out[env_id] = left_actions;
        right_actions_out[env_id] = right_actions;
    """
    try:
        return mx.fast.metal_kernel(
            name="biped_support_metrics",
            input_names=["contact_mask", "action_targets"],
            output_names=[
                "support_ratio_out",
                "left_support_out",
                "right_support_out",
                "left_actions_out",
                "right_actions_out",
            ],
            source=source,
        )
    except Exception:
        return None


def quadruped_support_metrics_hotpath(contact_mask: mx.array, action_targets: mx.array) -> tuple[mx.array, ...]:
    global _quadruped_support_metrics_metal
    if _quadruped_support_metrics_metal is None:
        _quadruped_support_metrics_metal = _build_quadruped_support_metrics_metal_kernel()
    contact_mask = mx.array(contact_mask, dtype=mx.float32)
    action_targets = mx.array(action_targets, dtype=mx.float32)
    if _quadruped_support_metrics_metal is None:
        return _quadruped_support_metrics_compiled(contact_mask.astype(mx.bool_), action_targets)
    if int(contact_mask.shape[0]) == 0:
        empty = mx.zeros((0,), dtype=mx.float32)
        return empty, empty, empty, empty, empty, empty, empty, empty, empty
    outputs = _quadruped_support_metrics_metal(
        inputs=[contact_mask, action_targets],
        grid=(int(contact_mask.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(int(contact_mask.shape[0]),)] * 9,
        output_dtypes=[mx.float32] * 9,
    )
    return tuple(outputs)


def biped_support_metrics_hotpath(contact_mask: mx.array, action_targets: mx.array) -> tuple[mx.array, ...]:
    global _biped_support_metrics_metal
    if _biped_support_metrics_metal is None:
        _biped_support_metrics_metal = _build_biped_support_metrics_metal_kernel()
    contact_mask = mx.array(contact_mask, dtype=mx.float32)
    action_targets = mx.array(action_targets, dtype=mx.float32)
    if _biped_support_metrics_metal is None:
        return _biped_support_metrics_compiled(contact_mask.astype(mx.bool_), action_targets)
    if int(contact_mask.shape[0]) == 0:
        empty = mx.zeros((0,), dtype=mx.float32)
        return empty, empty, empty, empty, empty
    outputs = _biped_support_metrics_metal(
        inputs=[contact_mask, action_targets],
        grid=(int(contact_mask.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(int(contact_mask.shape[0]),)] * 5,
        output_dtypes=[mx.float32] * 5,
    )
    return tuple(outputs)


def _anymal_leg_extension_impl(joint_pos: mx.array) -> mx.array:
    num_envs = joint_pos.shape[0]
    grouped = joint_pos.reshape((num_envs, 4, 3))
    hip_pitch = grouped[:, :, 1]
    knee = grouped[:, :, 2]
    extension = 0.20 + 0.16 * mx.cos(hip_pitch) + 0.18 * mx.cos(hip_pitch + knee)
    return mx.clip(extension, 0.22, 0.62)


_anymal_leg_extension_compiled = mx.compile(_anymal_leg_extension_impl)
_anymal_leg_extension_metal = None
_anymal_leg_extension_hotpath_initialized = False
ANYMAL_LEG_EXTENSION_HOTPATH_BACKEND = HOTPATH_BACKEND


def _build_anymal_leg_extension_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        uint base = env_id * 12;
        for (uint leg_id = 0; leg_id < 4; ++leg_id) {
            uint joint_base = base + leg_id * 3;
            float hip_pitch = joint_pos[joint_base + 1];
            float knee = joint_pos[joint_base + 2];
            float extension = 0.20f + 0.16f * metal::cos(hip_pitch) + 0.18f * metal::cos(hip_pitch + knee);
            extension_out[env_id * 4 + leg_id] = metal::clamp(extension, 0.22f, 0.62f);
        }
    """
    try:
        return mx.fast.metal_kernel(
            name="anymal_leg_extension",
            input_names=["joint_pos"],
            output_names=["extension_out"],
            source=source,
        )
    except Exception:
        return None


def _ensure_anymal_leg_extension_hotpath() -> None:
    global _anymal_leg_extension_metal
    global _anymal_leg_extension_hotpath_initialized
    global ANYMAL_LEG_EXTENSION_HOTPATH_BACKEND
    if _anymal_leg_extension_hotpath_initialized:
        return
    _anymal_leg_extension_hotpath_initialized = True
    _anymal_leg_extension_metal = _build_anymal_leg_extension_metal_kernel()
    if _anymal_leg_extension_metal is not None:
        ANYMAL_LEG_EXTENSION_HOTPATH_BACKEND = "mlx-metal-anymal-leg-extension"


def get_anymal_leg_extension_hotpath_backend() -> str:
    _ensure_anymal_leg_extension_hotpath()
    return ANYMAL_LEG_EXTENSION_HOTPATH_BACKEND


def anymal_leg_extension_hotpath(joint_pos: mx.array) -> mx.array:
    _ensure_anymal_leg_extension_hotpath()
    joint_pos = mx.array(joint_pos, dtype=mx.float32)
    if _anymal_leg_extension_metal is None:
        return _anymal_leg_extension_compiled(joint_pos)
    if int(joint_pos.shape[0]) == 0:
        return mx.zeros((0, 4), dtype=mx.float32)
    outputs = _anymal_leg_extension_metal(
        inputs=[joint_pos],
        grid=(int(joint_pos.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(int(joint_pos.shape[0]), 4)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


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
    extension = anymal_leg_extension_hotpath(joint_pos)
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


def _quat_from_euler_xyz_impl(roll: mx.array, pitch: mx.array, yaw: mx.array) -> mx.array:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = mx.cos(half_roll)
    sr = mx.sin(half_roll)
    cp = mx.cos(half_pitch)
    sp = mx.sin(half_pitch)
    cy = mx.cos(half_yaw)
    sy = mx.sin(half_yaw)
    quat = mx.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )
    return _quat_normalize(quat)


def _ur10e_end_effector_pose_impl(joint_pos: mx.array) -> tuple[mx.array, mx.array]:
    q0 = joint_pos[:, 0]
    q1 = joint_pos[:, 1]
    q2 = joint_pos[:, 2]
    q3 = joint_pos[:, 3]
    q4 = joint_pos[:, 4]
    q5 = joint_pos[:, 5]

    elbow = q1 + q2
    wrist = elbow + 0.5 * q3

    x = 0.72 + 0.22 * mx.cos(q1) + 0.18 * mx.cos(elbow) + 0.08 * mx.cos(wrist) - 0.06 * mx.sin(q0)
    y = -0.225 + 0.18 * mx.sin(q0) + 0.06 * mx.sin(q0 + 0.5 * q1) + 0.03 * mx.tanh(q5)
    z = 0.20 + 0.18 * mx.sin(-q1) + 0.11 * mx.sin(-elbow) + 0.05 * mx.sin(-wrist) - 0.03 * mx.tanh(q4)

    roll = math.pi + 0.20 * mx.tanh(q3)
    pitch = -0.10 * mx.tanh(q1 + q2) + 0.08 * mx.tanh(q4)
    yaw = -math.pi / 2.0 + q0 + 0.18 * mx.tanh(q5)
    quat = _quat_from_euler_xyz_impl(roll, pitch, yaw)
    return mx.stack((x, y, z), axis=-1).astype(mx.float32), quat.astype(mx.float32)


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


_franka_stack_object_step_compiled = mx.compile(_franka_stack_object_step_impl)
_franka_stack_object_step_metal = None
_franka_stack_object_hotpath_initialized = False
FRANKA_STACK_HOTPATH_BACKEND = HOTPATH_BACKEND
_franka_lift_object_step_metal = None
_franka_lift_hotpath_initialized = False
FRANKA_LIFT_HOTPATH_BACKEND = HOTPATH_BACKEND
_franka_stack_rgb_step_metal = None
_franka_stack_rgb_hotpath_initialized = False
FRANKA_STACK_RGB_HOTPATH_BACKEND = HOTPATH_BACKEND


def _build_franka_stack_object_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float physics_dt = params[0];
        float gripper_lower_limit = params[1];
        float gripper_upper_limit = params[2];
        float gripper_closed_threshold = params[3];
        float stack_release_open_threshold = params[4];
        float grasp_distance_threshold = params[5];
        float grasp_offset_z = params[6];
        float table_height = params[7];
        float stack_offset_z = params[8];
        float stack_xy_threshold = params[9];
        float stack_z_threshold = params[10];

        float3 cube = float3(cube_pos_w[env_id * 3 + 0], cube_pos_w[env_id * 3 + 1], cube_pos_w[env_id * 3 + 2]);
        float3 support = float3(
            support_cube_pos_w[env_id * 3 + 0],
            support_cube_pos_w[env_id * 3 + 1],
            support_cube_pos_w[env_id * 3 + 2]
        );
        float3 ee = float3(ee_pos_w[env_id * 3 + 0], ee_pos_w[env_id * 3 + 1], ee_pos_w[env_id * 3 + 2]);
        float gripper_joint = gripper_joint_pos[env_id];
        float gripper_action_value = gripper_action[env_id];
        float grasped_prev = grasped[env_id];
        float stacked_prev = stacked[env_id];

        float gripper_target = gripper_action_value < 0.0f ? gripper_lower_limit : gripper_upper_limit;
        float gripper_velocity = (gripper_target - gripper_joint) / physics_dt;
        float3 cube_to_ee = cube - ee;
        float dist_to_cube = metal::sqrt(cube_to_ee.x * cube_to_ee.x + cube_to_ee.y * cube_to_ee.y + cube_to_ee.z * cube_to_ee.z);
        float gripper_closed = gripper_target <= gripper_closed_threshold ? 1.0f : 0.0f;
        float gripper_open = gripper_target >= stack_release_open_threshold ? 1.0f : 0.0f;
        float can_grasp = dist_to_cube <= grasp_distance_threshold ? 1.0f : 0.0f;
        float next_grasped = ((grasped_prev > 0.5f || can_grasp > 0.5f) && gripper_closed > 0.5f && stacked_prev <= 0.5f)
            ? 1.0f
            : 0.0f;

        float3 attached_cube = ee + float3(0.0f, 0.0f, -grasp_offset_z);
        float3 stack_target = support + float3(0.0f, 0.0f, stack_offset_z);
        float3 release_cube = grasped_prev > 0.5f ? attached_cube : cube;
        float2 release_xy = float2(release_cube.x, release_cube.y);
        float2 stack_xy = float2(stack_target.x, stack_target.y);
        float2 stack_xy_delta = release_xy - stack_xy;
        float stack_xy_error = metal::sqrt(stack_xy_delta.x * stack_xy_delta.x + stack_xy_delta.y * stack_xy_delta.y);
        float stack_z_error = metal::fabs(release_cube.z - stack_target.z);
        float newly_stacked = (grasped_prev > 0.5f && gripper_open > 0.5f && stack_xy_error <= stack_xy_threshold && stack_z_error <= stack_z_threshold)
            ? 1.0f
            : 0.0f;
        float next_stacked = stacked_prev > 0.5f || newly_stacked > 0.5f ? 1.0f : 0.0f;
        next_grasped = next_grasped * (1.0f - next_stacked);

        float3 resting_cube = float3(
            release_cube.x,
            release_cube.y,
            metal::max(table_height, release_cube.z - physics_dt * 0.35f)
        );
        float3 next_cube = resting_cube;
        if (next_stacked > 0.5f) {
            next_cube = stack_target;
        } else if (next_grasped > 0.5f) {
            next_cube = attached_cube;
        }

        gripper_target_out[env_id] = gripper_target;
        gripper_velocity_out[env_id] = gripper_velocity;
        next_grasped_out[env_id] = next_grasped;
        next_stacked_out[env_id] = next_stacked;
        next_cube_pos_w[env_id * 3 + 0] = next_cube.x;
        next_cube_pos_w[env_id * 3 + 1] = next_cube.y;
        next_cube_pos_w[env_id * 3 + 2] = next_cube.z;
    """
    try:
        return mx.fast.metal_kernel(
            name="franka_stack_object_step",
            input_names=[
                "cube_pos_w",
                "support_cube_pos_w",
                "ee_pos_w",
                "gripper_joint_pos",
                "gripper_action",
                "grasped",
                "stacked",
                "params",
            ],
            output_names=[
                "gripper_target_out",
                "gripper_velocity_out",
                "next_grasped_out",
                "next_stacked_out",
                "next_cube_pos_w",
            ],
            source=source,
        )
    except Exception:
        return None


def _build_franka_lift_object_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float physics_dt = params[0];
        float gripper_lower_limit = params[1];
        float gripper_upper_limit = params[2];
        float gripper_closed_threshold = params[3];
        float grasp_distance_threshold = params[4];
        float grasp_offset_z = params[5];
        float table_height = params[6];

        float3 cube = float3(cube_pos_w[env_id * 3 + 0], cube_pos_w[env_id * 3 + 1], cube_pos_w[env_id * 3 + 2]);
        float3 ee = float3(ee_pos_w[env_id * 3 + 0], ee_pos_w[env_id * 3 + 1], ee_pos_w[env_id * 3 + 2]);
        float gripper_joint = gripper_joint_pos[env_id];
        float gripper_action_value = gripper_action[env_id];
        float grasped_prev = grasped[env_id];

        float gripper_target = gripper_action_value < 0.0f ? gripper_lower_limit : gripper_upper_limit;
        float gripper_velocity = (gripper_target - gripper_joint) / physics_dt;

        float3 diff = cube - ee;
        float dist_to_cube = metal::length(diff);
        bool gripper_closed = gripper_target <= gripper_closed_threshold;
        bool can_grasp = dist_to_cube <= grasp_distance_threshold;
        bool next_grasped = (grasped_prev > 0.5f || (can_grasp && gripper_closed)) && gripper_closed;

        float3 attached_cube = ee + float3(0.0f, 0.0f, -grasp_offset_z);
        float resting_z = metal::max(table_height, cube.z - physics_dt * 0.35f);
        float3 resting_cube = float3(cube.x, cube.y, resting_z);
        float3 next_cube = next_grasped ? attached_cube : resting_cube;

        gripper_target_out[env_id] = gripper_target;
        gripper_velocity_out[env_id] = gripper_velocity;
        next_grasped_out[env_id] = next_grasped ? 1.0f : 0.0f;
        next_cube_pos_out[env_id * 3 + 0] = next_cube.x;
        next_cube_pos_out[env_id * 3 + 1] = next_cube.y;
        next_cube_pos_out[env_id * 3 + 2] = next_cube.z;
    """
    try:
        return mx.fast.metal_kernel(
            name="franka_lift_object_step",
            input_names=["cube_pos_w", "ee_pos_w", "gripper_joint_pos", "gripper_action", "grasped", "params"],
            output_names=["gripper_target_out", "gripper_velocity_out", "next_grasped_out", "next_cube_pos_out"],
            source=source,
        )
    except Exception:
        return None


def _ensure_franka_stack_object_hotpath() -> None:
    global _franka_stack_object_step_metal
    global _franka_stack_object_hotpath_initialized
    global FRANKA_STACK_HOTPATH_BACKEND
    if _franka_stack_object_hotpath_initialized:
        return
    _franka_stack_object_hotpath_initialized = True
    _franka_stack_object_step_metal = _build_franka_stack_object_metal_kernel()
    if _franka_stack_object_step_metal is not None:
        FRANKA_STACK_HOTPATH_BACKEND = "mlx-metal-franka-stack"


def get_franka_stack_hotpath_backend() -> str:
    _ensure_franka_stack_object_hotpath()
    return FRANKA_STACK_HOTPATH_BACKEND


def _ensure_franka_lift_hotpath() -> None:
    global _franka_lift_object_step_metal
    global _franka_lift_hotpath_initialized
    global FRANKA_LIFT_HOTPATH_BACKEND
    if _franka_lift_hotpath_initialized:
        return
    _franka_lift_hotpath_initialized = True
    _franka_lift_object_step_metal = _build_franka_lift_object_metal_kernel()
    if _franka_lift_object_step_metal is not None:
        FRANKA_LIFT_HOTPATH_BACKEND = "mlx-metal-franka-lift"


def get_franka_lift_hotpath_backend() -> str:
    _ensure_franka_lift_hotpath()
    return FRANKA_LIFT_HOTPATH_BACKEND


def _build_franka_stack_rgb_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float physics_dt = params[0];
        float gripper_lower_limit = params[1];
        float gripper_upper_limit = params[2];
        float gripper_closed_threshold = params[3];
        float stack_release_open_threshold = params[4];
        float grasp_distance_threshold = params[5];
        float grasp_offset_z = params[6];
        float table_height = params[7];
        float stack_offset_z = params[8];
        float stack_xy_threshold = params[9];
        float stack_z_threshold = params[10];

        float3 middle_cube = float3(
            middle_cube_pos_w[env_id * 3 + 0],
            middle_cube_pos_w[env_id * 3 + 1],
            middle_cube_pos_w[env_id * 3 + 2]
        );
        float3 top_cube = float3(
            top_cube_pos_w[env_id * 3 + 0],
            top_cube_pos_w[env_id * 3 + 1],
            top_cube_pos_w[env_id * 3 + 2]
        );
        float3 support_cube = float3(
            support_cube_pos_w[env_id * 3 + 0],
            support_cube_pos_w[env_id * 3 + 1],
            support_cube_pos_w[env_id * 3 + 2]
        );
        float3 ee = float3(ee_pos_w[env_id * 3 + 0], ee_pos_w[env_id * 3 + 1], ee_pos_w[env_id * 3 + 2]);
        float gripper_joint = gripper_joint_pos[env_id];
        float gripper_action_value = gripper_action[env_id];
        float middle_grasped_prev = middle_grasped[env_id];
        float top_grasped_prev = top_grasped[env_id];
        float middle_stacked_prev = middle_stacked[env_id];
        float top_stacked_prev = top_stacked[env_id];

        bool use_top_cube = middle_stacked_prev > 0.5f;
        float3 active_cube = use_top_cube ? top_cube : middle_cube;
        float3 active_target = use_top_cube ? (middle_cube + float3(0.0f, 0.0f, stack_offset_z)) : (support_cube + float3(0.0f, 0.0f, stack_offset_z));
        float active_grasped_prev = use_top_cube ? top_grasped_prev : middle_grasped_prev;

        float gripper_target = gripper_action_value < 0.0f ? gripper_lower_limit : gripper_upper_limit;
        float gripper_velocity = (gripper_target - gripper_joint) / physics_dt;
        float3 cube_to_ee = active_cube - ee;
        float dist_to_cube = metal::sqrt(cube_to_ee.x * cube_to_ee.x + cube_to_ee.y * cube_to_ee.y + cube_to_ee.z * cube_to_ee.z);
        float gripper_closed = gripper_target <= gripper_closed_threshold ? 1.0f : 0.0f;
        float gripper_open = gripper_target >= stack_release_open_threshold ? 1.0f : 0.0f;
        float can_grasp = dist_to_cube <= grasp_distance_threshold ? 1.0f : 0.0f;
        float next_active_grasped = ((active_grasped_prev > 0.5f || can_grasp > 0.5f) && gripper_closed > 0.5f && top_stacked_prev <= 0.5f)
            ? 1.0f
            : 0.0f;

        float3 attached_cube = ee + float3(0.0f, 0.0f, -grasp_offset_z);
        float3 release_cube = active_grasped_prev > 0.5f ? attached_cube : active_cube;
        float2 release_xy = float2(release_cube.x, release_cube.y);
        float2 active_target_xy = float2(active_target.x, active_target.y);
        float2 stack_xy_delta = release_xy - active_target_xy;
        float stack_xy_error = metal::sqrt(stack_xy_delta.x * stack_xy_delta.x + stack_xy_delta.y * stack_xy_delta.y);
        float stack_z_error = metal::fabs(release_cube.z - active_target.z);
        float newly_stacked = (active_grasped_prev > 0.5f && gripper_open > 0.5f && stack_xy_error <= stack_xy_threshold && stack_z_error <= stack_z_threshold)
            ? 1.0f
            : 0.0f;

        float next_middle_stacked = middle_stacked_prev > 0.5f || (!use_top_cube && newly_stacked > 0.5f) ? 1.0f : 0.0f;
        float next_top_stacked = top_stacked_prev > 0.5f || (use_top_cube && newly_stacked > 0.5f) ? 1.0f : 0.0f;

        float3 resting_cube = float3(
            release_cube.x,
            release_cube.y,
            metal::max(table_height, release_cube.z - physics_dt * 0.35f)
        );
        float3 next_active_cube = resting_cube;
        if (newly_stacked > 0.5f) {
            next_active_cube = active_target;
        } else if (next_active_grasped > 0.5f) {
            next_active_cube = attached_cube;
        }

        float3 next_middle_cube = use_top_cube ? middle_cube : next_active_cube;
        float3 next_top_cube = use_top_cube ? next_active_cube : top_cube;

        float next_middle_grasped = use_top_cube ? middle_grasped_prev : (next_active_grasped * (1.0f - next_middle_stacked));
        float next_top_grasped = use_top_cube ? (next_active_grasped * (1.0f - next_top_stacked)) : top_grasped_prev;

        gripper_target_out[env_id] = gripper_target;
        gripper_velocity_out[env_id] = gripper_velocity;
        next_middle_grasped_out[env_id] = next_middle_grasped;
        next_top_grasped_out[env_id] = next_top_grasped;
        next_middle_stacked_out[env_id] = next_middle_stacked;
        next_top_stacked_out[env_id] = next_top_stacked;
        next_middle_cube_pos_w[env_id * 3 + 0] = next_middle_cube.x;
        next_middle_cube_pos_w[env_id * 3 + 1] = next_middle_cube.y;
        next_middle_cube_pos_w[env_id * 3 + 2] = next_middle_cube.z;
        next_top_cube_pos_w[env_id * 3 + 0] = next_top_cube.x;
        next_top_cube_pos_w[env_id * 3 + 1] = next_top_cube.y;
        next_top_cube_pos_w[env_id * 3 + 2] = next_top_cube.z;
    """
    try:
        return mx.fast.metal_kernel(
            name="franka_stack_rgb_step",
            input_names=[
                "middle_cube_pos_w",
                "top_cube_pos_w",
                "support_cube_pos_w",
                "ee_pos_w",
                "gripper_joint_pos",
                "gripper_action",
                "middle_grasped",
                "top_grasped",
                "middle_stacked",
                "top_stacked",
                "params",
            ],
            output_names=[
                "gripper_target_out",
                "gripper_velocity_out",
                "next_middle_grasped_out",
                "next_top_grasped_out",
                "next_middle_stacked_out",
                "next_top_stacked_out",
                "next_middle_cube_pos_w",
                "next_top_cube_pos_w",
            ],
            source=source,
        )
    except Exception:
        return None


def _ensure_franka_stack_rgb_hotpath() -> None:
    global _franka_stack_rgb_step_metal
    global _franka_stack_rgb_hotpath_initialized
    global FRANKA_STACK_RGB_HOTPATH_BACKEND
    if _franka_stack_rgb_hotpath_initialized:
        return
    _franka_stack_rgb_hotpath_initialized = True
    _franka_stack_rgb_step_metal = _build_franka_stack_rgb_metal_kernel()
    if _franka_stack_rgb_step_metal is not None:
        FRANKA_STACK_RGB_HOTPATH_BACKEND = "mlx-metal-franka-stack-rgb"


def get_franka_stack_rgb_hotpath_backend() -> str:
    _ensure_franka_stack_rgb_hotpath()
    return FRANKA_STACK_RGB_HOTPATH_BACKEND


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

_franka_stack_rgb_step_compiled = mx.compile(_franka_stack_rgb_step_impl)
_franka_end_effector_position_compiled = mx.compile(_franka_end_effector_position_impl)
_ur10e_end_effector_pose_compiled = mx.compile(_ur10e_end_effector_pose_impl)
_franka_end_effector_position_metal = None
_franka_end_effector_hotpath_initialized = False
FRANKA_HOTPATH_BACKEND = HOTPATH_BACKEND
UR10E_HOTPATH_BACKEND = HOTPATH_BACKEND
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


def get_ur10e_hotpath_backend() -> str:
    return UR10E_HOTPATH_BACKEND


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


def ur10e_end_effector_pose_hotpath(joint_pos: mx.array) -> tuple[mx.array, mx.array]:
    joint_pos = mx.array(joint_pos, dtype=mx.float32)
    if int(joint_pos.shape[0]) == 0:
        return mx.zeros((0, 3), dtype=mx.float32), mx.zeros((0, 4), dtype=mx.float32)
    return _ur10e_end_effector_pose_compiled(joint_pos)


_franka_cabinet_step_compiled = mx.compile(_franka_cabinet_step_impl)
_franka_cabinet_step_metal = None
_franka_cabinet_hotpath_initialized = False
FRANKA_CABINET_HOTPATH_BACKEND = HOTPATH_BACKEND


def _build_franka_cabinet_metal_kernel():
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        return None
    source = r"""
        uint env_id = thread_position_in_grid.x;
        float physics_dt = params[0];
        float gripper_lower_limit = params[1];
        float gripper_upper_limit = params[2];
        float gripper_closed_threshold = params[3];
        float handle_grasp_threshold = params[4];
        float drawer_open_distance_max = params[5];
        float drawer_success_distance = params[6];

        float3 handle_anchor = float3(
            handle_anchor_pos_w[env_id * 3 + 0],
            handle_anchor_pos_w[env_id * 3 + 1],
            handle_anchor_pos_w[env_id * 3 + 2]
        );
        float3 ee = float3(ee_pos_w[env_id * 3 + 0], ee_pos_w[env_id * 3 + 1], ee_pos_w[env_id * 3 + 2]);
        float gripper_joint = gripper_joint_pos[env_id];
        float gripper_action_value = gripper_action[env_id];
        float grasped_prev = grasped[env_id];
        float opened_prev = opened[env_id];
        float drawer_open_prev = drawer_open_amount[env_id];

        float gripper_target = gripper_action_value < 0.0f ? gripper_lower_limit : gripper_upper_limit;
        float gripper_velocity = (gripper_target - gripper_joint) / physics_dt;
        float3 current_handle = float3(handle_anchor.x + drawer_open_prev, handle_anchor.y, handle_anchor.z);
        float3 handle_delta = current_handle - ee;
        float dist_to_handle = metal::sqrt(
            handle_delta.x * handle_delta.x + handle_delta.y * handle_delta.y + handle_delta.z * handle_delta.z
        );
        float gripper_closed = gripper_target <= gripper_closed_threshold ? 1.0f : 0.0f;
        float can_grasp = dist_to_handle <= handle_grasp_threshold ? 1.0f : 0.0f;
        float next_grasped = ((grasped_prev > 0.5f || can_grasp > 0.5f) && gripper_closed > 0.5f && opened_prev <= 0.5f)
            ? 1.0f
            : 0.0f;

        float pulled_open = metal::clamp(ee.x - handle_anchor.x, 0.0f, drawer_open_distance_max);
        float next_drawer_open = next_grasped > 0.5f ? pulled_open : drawer_open_prev;
        float next_opened = (opened_prev > 0.5f || next_drawer_open >= drawer_success_distance) ? 1.0f : 0.0f;
        next_grasped = next_grasped * (1.0f - next_opened);

        float3 next_handle = float3(handle_anchor.x + next_drawer_open, handle_anchor.y, handle_anchor.z);

        gripper_target_out[env_id] = gripper_target;
        gripper_velocity_out[env_id] = gripper_velocity;
        next_grasped_out[env_id] = next_grasped;
        next_opened_out[env_id] = next_opened;
        next_drawer_open_amount_out[env_id] = next_drawer_open;
        next_handle_pos_w[env_id * 3 + 0] = next_handle.x;
        next_handle_pos_w[env_id * 3 + 1] = next_handle.y;
        next_handle_pos_w[env_id * 3 + 2] = next_handle.z;
    """
    try:
        return mx.fast.metal_kernel(
            name="franka_cabinet_step",
            input_names=[
                "handle_anchor_pos_w",
                "ee_pos_w",
                "gripper_joint_pos",
                "gripper_action",
                "grasped",
                "opened",
                "drawer_open_amount",
                "params",
            ],
            output_names=[
                "gripper_target_out",
                "gripper_velocity_out",
                "next_grasped_out",
                "next_opened_out",
                "next_drawer_open_amount_out",
                "next_handle_pos_w",
            ],
            source=source,
        )
    except Exception:
        return None


def _ensure_franka_cabinet_hotpath() -> None:
    global _franka_cabinet_step_metal
    global _franka_cabinet_hotpath_initialized
    global FRANKA_CABINET_HOTPATH_BACKEND
    if _franka_cabinet_hotpath_initialized:
        return
    _franka_cabinet_hotpath_initialized = True
    _franka_cabinet_step_metal = _build_franka_cabinet_metal_kernel()
    if _franka_cabinet_step_metal is not None:
        FRANKA_CABINET_HOTPATH_BACKEND = "mlx-metal-franka-cabinet"


def get_franka_cabinet_hotpath_backend() -> str:
    _ensure_franka_cabinet_hotpath()
    return FRANKA_CABINET_HOTPATH_BACKEND


def franka_cabinet_step_hotpath(
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
    _ensure_franka_cabinet_hotpath()
    handle_anchor_pos_w = mx.array(handle_anchor_pos_w, dtype=mx.float32)
    ee_pos_w = mx.array(ee_pos_w, dtype=mx.float32)
    gripper_joint_pos = mx.array(gripper_joint_pos, dtype=mx.float32)
    gripper_action = mx.array(gripper_action, dtype=mx.float32)
    grasped = mx.array(grasped, dtype=mx.bool_)
    opened = mx.array(opened, dtype=mx.bool_)
    drawer_open_amount = mx.array(drawer_open_amount, dtype=mx.float32)
    if _franka_cabinet_step_metal is None:
        return _franka_cabinet_step_compiled(
            handle_anchor_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            grasped,
            opened,
            drawer_open_amount,
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            handle_grasp_threshold,
            drawer_open_distance_max,
            drawer_success_distance,
        )
    if int(handle_anchor_pos_w.shape[0]) == 0:
        return (
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0, 3), dtype=mx.float32),
        )
    params = mx.array(
        [
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            handle_grasp_threshold,
            drawer_open_distance_max,
            drawer_success_distance,
        ],
        dtype=mx.float32,
    )
    outputs = _franka_cabinet_step_metal(
        inputs=[
            handle_anchor_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            grasped,
            opened,
            drawer_open_amount,
            params,
        ],
        grid=(int(handle_anchor_pos_w.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[
            (int(handle_anchor_pos_w.shape[0]),),
            (int(handle_anchor_pos_w.shape[0]),),
            (int(handle_anchor_pos_w.shape[0]),),
            (int(handle_anchor_pos_w.shape[0]),),
            (int(handle_anchor_pos_w.shape[0]),),
            tuple(handle_anchor_pos_w.shape),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return (
        outputs[0],
        outputs[1],
        outputs[2].astype(mx.bool_),
        outputs[3].astype(mx.bool_),
        outputs[4],
        outputs[5],
    )


def franka_lift_object_step_hotpath(
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
    _ensure_franka_lift_hotpath()
    cube_pos_w = mx.array(cube_pos_w, dtype=mx.float32)
    ee_pos_w = mx.array(ee_pos_w, dtype=mx.float32)
    gripper_joint_pos = mx.array(gripper_joint_pos, dtype=mx.float32)
    gripper_action = mx.array(gripper_action, dtype=mx.float32)
    grasped = mx.array(grasped, dtype=mx.float32)
    if _franka_lift_object_step_metal is None:
        return _franka_lift_object_step_impl(
            cube_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            grasped.astype(mx.bool_),
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            grasp_distance_threshold,
            grasp_offset_z,
            table_height,
        )
    if int(cube_pos_w.shape[0]) == 0:
        return (
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0, 3), dtype=mx.float32),
        )
    params = mx.array(
        [
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            grasp_distance_threshold,
            grasp_offset_z,
            table_height,
        ],
        dtype=mx.float32,
    )
    outputs = _franka_lift_object_step_metal(
        inputs=[cube_pos_w, ee_pos_w, gripper_joint_pos, gripper_action, grasped, params],
        grid=(int(cube_pos_w.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[
            (int(cube_pos_w.shape[0]),),
            (int(cube_pos_w.shape[0]),),
            (int(cube_pos_w.shape[0]),),
            tuple(cube_pos_w.shape),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return outputs[0], outputs[1], outputs[2].astype(mx.bool_), outputs[3]


def franka_stack_rgb_step_hotpath(
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
    _ensure_franka_stack_rgb_hotpath()
    middle_cube_pos_w = mx.array(middle_cube_pos_w, dtype=mx.float32)
    top_cube_pos_w = mx.array(top_cube_pos_w, dtype=mx.float32)
    support_cube_pos_w = mx.array(support_cube_pos_w, dtype=mx.float32)
    ee_pos_w = mx.array(ee_pos_w, dtype=mx.float32)
    gripper_joint_pos = mx.array(gripper_joint_pos, dtype=mx.float32)
    gripper_action = mx.array(gripper_action, dtype=mx.float32)
    middle_grasped = mx.array(middle_grasped, dtype=mx.float32)
    top_grasped = mx.array(top_grasped, dtype=mx.float32)
    middle_stacked = mx.array(middle_stacked, dtype=mx.float32)
    top_stacked = mx.array(top_stacked, dtype=mx.float32)
    if _franka_stack_rgb_step_metal is None:
        return _franka_stack_rgb_step_compiled(
            middle_cube_pos_w,
            top_cube_pos_w,
            support_cube_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            middle_grasped.astype(mx.bool_),
            top_grasped.astype(mx.bool_),
            middle_stacked.astype(mx.bool_),
            top_stacked.astype(mx.bool_),
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            stack_release_open_threshold,
            grasp_distance_threshold,
            grasp_offset_z,
            table_height,
            stack_offset_z,
            stack_xy_threshold,
            stack_z_threshold,
        )
    if int(middle_cube_pos_w.shape[0]) == 0:
        return (
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0, 3), dtype=mx.float32),
            mx.zeros((0, 3), dtype=mx.float32),
        )
    params = mx.array(
        [
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            stack_release_open_threshold,
            grasp_distance_threshold,
            grasp_offset_z,
            table_height,
            stack_offset_z,
            stack_xy_threshold,
            stack_z_threshold,
        ],
        dtype=mx.float32,
    )
    outputs = _franka_stack_rgb_step_metal(
        inputs=[
            middle_cube_pos_w,
            top_cube_pos_w,
            support_cube_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            middle_grasped,
            top_grasped,
            middle_stacked,
            top_stacked,
            params,
        ],
        grid=(int(middle_cube_pos_w.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[
            (int(middle_cube_pos_w.shape[0]),),
            (int(middle_cube_pos_w.shape[0]),),
            (int(middle_cube_pos_w.shape[0]),),
            (int(middle_cube_pos_w.shape[0]),),
            (int(middle_cube_pos_w.shape[0]),),
            (int(middle_cube_pos_w.shape[0]),),
            tuple(middle_cube_pos_w.shape),
            tuple(top_cube_pos_w.shape),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return (
        outputs[0],
        outputs[1],
        outputs[2].astype(mx.bool_),
        outputs[3].astype(mx.bool_),
        outputs[4].astype(mx.bool_),
        outputs[5].astype(mx.bool_),
        outputs[6],
        outputs[7],
    )


def franka_stack_object_step_hotpath(
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
    _ensure_franka_stack_object_hotpath()
    cube_pos_w = mx.array(cube_pos_w, dtype=mx.float32)
    support_cube_pos_w = mx.array(support_cube_pos_w, dtype=mx.float32)
    ee_pos_w = mx.array(ee_pos_w, dtype=mx.float32)
    gripper_joint_pos = mx.array(gripper_joint_pos, dtype=mx.float32)
    gripper_action = mx.array(gripper_action, dtype=mx.float32)
    grasped = mx.array(grasped, dtype=mx.float32)
    stacked = mx.array(stacked, dtype=mx.float32)
    if _franka_stack_object_step_metal is None:
        return _franka_stack_object_step_compiled(
            cube_pos_w,
            support_cube_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            grasped.astype(mx.bool_),
            stacked.astype(mx.bool_),
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            stack_release_open_threshold,
            grasp_distance_threshold,
            grasp_offset_z,
            table_height,
            stack_offset_z,
            stack_xy_threshold,
            stack_z_threshold,
        )
    if int(cube_pos_w.shape[0]) == 0:
        return (
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0,), dtype=mx.bool_),
            mx.zeros((0, 3), dtype=mx.float32),
        )
    params = mx.array(
        [
            physics_dt,
            gripper_lower_limit,
            gripper_upper_limit,
            gripper_closed_threshold,
            stack_release_open_threshold,
            grasp_distance_threshold,
            grasp_offset_z,
            table_height,
            stack_offset_z,
            stack_xy_threshold,
            stack_z_threshold,
        ],
        dtype=mx.float32,
    )
    outputs = _franka_stack_object_step_metal(
        inputs=[
            cube_pos_w,
            support_cube_pos_w,
            ee_pos_w,
            gripper_joint_pos,
            gripper_action,
            grasped,
            stacked,
            params,
        ],
        grid=(int(cube_pos_w.shape[0]), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[
            (int(cube_pos_w.shape[0]),),
            (int(cube_pos_w.shape[0]),),
            (int(cube_pos_w.shape[0]),),
            (int(cube_pos_w.shape[0]),),
            tuple(cube_pos_w.shape),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
    )
    return outputs[0], outputs[1], outputs[2].astype(mx.bool_), outputs[3].astype(mx.bool_), outputs[4]


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
