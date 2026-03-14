# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the MLX helper-kernel compatibility layer."""

from __future__ import annotations

import math

import numpy as np

from isaaclab.backends.kernel_compat import (
    add_forces_and_torques_at_position_mlx,
    detect_cpu_fallback,
    reshape_tiled_image_mlx,
    set_forces_and_torques_at_position_mlx,
)
from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([-quat[..., :3], quat[..., 3:4]], axis=-1)


def _quat_rotate_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    cross_1 = np.cross(quat_xyz, vector, axis=-1)
    cross_2 = np.cross(quat_xyz, cross_1, axis=-1)
    return vector + 2.0 * (quat_w * cross_1 + cross_2)


def _quat_rotate_inv_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return _quat_rotate_xyzw(_quat_conjugate_xyzw(quat), vector)


def _reference_wrench_update(
    *,
    mode: str,
    env_ids: np.ndarray,
    body_ids: np.ndarray,
    forces: np.ndarray | None,
    torques: np.ndarray | None,
    positions: np.ndarray | None,
    link_positions: np.ndarray,
    link_quaternions: np.ndarray,
    composed_forces: np.ndarray,
    composed_torques: np.ndarray,
    is_global: bool,
) -> tuple[np.ndarray, np.ndarray]:
    force_out = composed_forces.copy()
    torque_out = composed_torques.copy()

    for env_offset, env_id in enumerate(env_ids):
        for body_offset, body_id in enumerate(body_ids):
            link_position = link_positions[env_id, body_id]
            link_quaternion = link_quaternions[env_id, body_id]

            if torques is not None:
                torque_value = torques[env_offset, body_offset]
                torque_link = _quat_rotate_inv_xyzw(link_quaternion, torque_value) if is_global else torque_value
                if mode == "add":
                    torque_out[env_id, body_id] += torque_link
                else:
                    torque_out[env_id, body_id] = torque_link

            if forces is not None:
                force_value = forces[env_offset, body_offset]
                force_link = _quat_rotate_inv_xyzw(link_quaternion, force_value) if is_global else force_value
                if mode == "add":
                    force_out[env_id, body_id] += force_link
                else:
                    force_out[env_id, body_id] = force_link

                if positions is not None:
                    position_value = positions[env_offset, body_offset]
                    position_link = position_value - link_position if is_global else position_value
                    torque_from_force = np.cross(position_link, force_link)
                    if mode == "add":
                        torque_out[env_id, body_id] += torque_from_force
                    else:
                        torque_out[env_id, body_id] = torque_from_force

    return force_out, torque_out


def test_reshape_tiled_image_mlx_matches_reference_for_partial_grid():
    image_height = 2
    image_width = 3
    num_channels = 2
    num_tiles_x = 2
    num_cameras = 3
    num_tiles_y = math.ceil(num_cameras / num_tiles_x)
    tiled = np.arange(
        num_tiles_y * image_height * num_tiles_x * image_width * num_channels,
        dtype=np.float32,
    )
    expected = tiled.reshape(num_tiles_y, image_height, num_tiles_x, image_width, num_channels)
    expected = expected.transpose(0, 2, 1, 3, 4).reshape(num_tiles_y * num_tiles_x, image_height, image_width, num_channels)
    expected = expected[:num_cameras]

    reshaped = reshape_tiled_image_mlx(
        mx.array(tiled),
        num_cameras=num_cameras,
        image_height=image_height,
        image_width=image_width,
        num_channels=num_channels,
        num_tiles_x=num_tiles_x,
    )
    mx.eval(reshaped)

    assert np.allclose(np.array(reshaped), expected)


def test_add_forces_and_torques_at_position_mlx_matches_local_reference():
    env_ids = np.array([0, 1], dtype=np.int32)
    body_ids = np.array([0, 2], dtype=np.int32)
    composed_forces = np.zeros((2, 3, 3), dtype=np.float32)
    composed_torques = np.zeros((2, 3, 3), dtype=np.float32)
    link_positions = np.zeros((2, 3, 3), dtype=np.float32)
    link_quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 3, 1))
    forces = np.array(
        [
            [[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]],
            [[-1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    torques = np.array(
        [
            [[0.1, 0.2, 0.3], [0.0, 0.5, -0.5]],
            [[-0.5, 0.25, 0.75], [1.0, -1.0, 0.25]],
        ],
        dtype=np.float32,
    )
    positions = np.array(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    expected_forces, expected_torques = _reference_wrench_update(
        mode="add",
        env_ids=env_ids,
        body_ids=body_ids,
        forces=forces,
        torques=torques,
        positions=positions,
        link_positions=link_positions,
        link_quaternions=link_quaternions,
        composed_forces=composed_forces,
        composed_torques=composed_torques,
        is_global=False,
    )

    actual_forces, actual_torques = add_forces_and_torques_at_position_mlx(
        mx.array(env_ids),
        mx.array(body_ids),
        mx.array(forces),
        mx.array(torques),
        mx.array(positions),
        mx.array(link_positions),
        mx.array(link_quaternions),
        mx.array(composed_forces),
        mx.array(composed_torques),
        is_global=False,
    )
    mx.eval(actual_forces, actual_torques)

    assert np.allclose(np.array(actual_forces), expected_forces)
    assert np.allclose(np.array(actual_torques), expected_torques)


def test_add_forces_and_torques_at_position_mlx_matches_global_reference():
    env_ids = np.array([0], dtype=np.int32)
    body_ids = np.array([1], dtype=np.int32)
    composed_forces = np.zeros((1, 2, 3), dtype=np.float32)
    composed_torques = np.zeros((1, 2, 3), dtype=np.float32)
    link_positions = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]], dtype=np.float32)
    quarter_turn_z = np.array([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float32)
    link_quaternions = np.array([[[0.0, 0.0, 0.0, 1.0], quarter_turn_z]], dtype=np.float32)
    forces = np.array([[[2.0, 0.0, 0.0]]], dtype=np.float32)
    torques = np.array([[[0.0, 3.0, 0.0]]], dtype=np.float32)
    positions = np.array([[[2.0, 4.0, 0.0]]], dtype=np.float32)

    expected_forces, expected_torques = _reference_wrench_update(
        mode="add",
        env_ids=env_ids,
        body_ids=body_ids,
        forces=forces,
        torques=torques,
        positions=positions,
        link_positions=link_positions,
        link_quaternions=link_quaternions,
        composed_forces=composed_forces,
        composed_torques=composed_torques,
        is_global=True,
    )

    actual_forces, actual_torques = add_forces_and_torques_at_position_mlx(
        mx.array(env_ids),
        mx.array(body_ids),
        mx.array(forces),
        mx.array(torques),
        mx.array(positions),
        mx.array(link_positions),
        mx.array(link_quaternions),
        mx.array(composed_forces),
        mx.array(composed_torques),
        is_global=True,
    )
    mx.eval(actual_forces, actual_torques)

    assert np.allclose(np.array(actual_forces), expected_forces)
    assert np.allclose(np.array(actual_torques), expected_torques)


def test_set_forces_and_torques_at_position_mlx_preserves_overwrite_semantics():
    env_ids = np.array([0], dtype=np.int32)
    body_ids = np.array([0], dtype=np.int32)
    composed_forces = np.array([[[5.0, 5.0, 5.0]]], dtype=np.float32)
    composed_torques = np.array([[[7.0, 7.0, 7.0]]], dtype=np.float32)
    link_positions = np.zeros((1, 1, 3), dtype=np.float32)
    link_quaternions = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
    forces = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    torques = np.array([[[0.0, 2.0, 0.0]]], dtype=np.float32)
    positions = np.array([[[0.0, 3.0, 0.0]]], dtype=np.float32)

    expected_forces, expected_torques = _reference_wrench_update(
        mode="set",
        env_ids=env_ids,
        body_ids=body_ids,
        forces=forces,
        torques=torques,
        positions=positions,
        link_positions=link_positions,
        link_quaternions=link_quaternions,
        composed_forces=composed_forces,
        composed_torques=composed_torques,
        is_global=False,
    )

    actual_forces, actual_torques = set_forces_and_torques_at_position_mlx(
        mx.array(env_ids),
        mx.array(body_ids),
        mx.array(forces),
        mx.array(torques),
        mx.array(positions),
        mx.array(link_positions),
        mx.array(link_quaternions),
        mx.array(composed_forces),
        mx.array(composed_torques),
        is_global=False,
    )
    mx.eval(actual_forces, actual_torques)

    assert np.allclose(np.array(actual_forces), expected_forces)
    assert np.allclose(np.array(actual_torques), expected_torques)
    assert np.allclose(np.array(actual_torques)[0, 0], np.array([0.0, 0.0, -3.0], dtype=np.float32))


def test_set_forces_and_torques_at_position_mlx_leaves_untouched_channels_when_omitted():
    env_ids = np.array([0], dtype=np.int32)
    body_ids = np.array([0], dtype=np.int32)
    composed_forces = np.array([[[3.0, 4.0, 5.0]]], dtype=np.float32)
    composed_torques = np.array([[[6.0, 7.0, 8.0]]], dtype=np.float32)
    link_positions = np.zeros((1, 1, 3), dtype=np.float32)
    link_quaternions = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
    forces = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)

    actual_forces, actual_torques = set_forces_and_torques_at_position_mlx(
        mx.array(env_ids),
        mx.array(body_ids),
        mx.array(forces),
        None,
        None,
        mx.array(link_positions),
        mx.array(link_quaternions),
        mx.array(composed_forces),
        mx.array(composed_torques),
        is_global=False,
    )
    mx.eval(actual_forces, actual_torques)

    assert np.allclose(np.array(actual_forces)[0, 0], forces[0, 0])
    assert np.allclose(np.array(actual_torques)[0, 0], composed_torques[0, 0])


def test_detect_cpu_fallback_flags_cpu_kernel_backend_only():
    cpu_state = {"kernel_backend": "cpu"}
    metal_state = {"kernel_backend": "metal", "device": "cpu"}

    assert detect_cpu_fallback(cpu_state) == {
        "detected": True,
        "reason": "kernel-backend=cpu",
        "active_kernel_backend": "cpu",
        "expected_kernel_backend": "metal",
    }
    assert detect_cpu_fallback(metal_state) == {
        "detected": False,
        "reason": None,
        "active_kernel_backend": "metal",
        "expected_kernel_backend": "metal",
    }
