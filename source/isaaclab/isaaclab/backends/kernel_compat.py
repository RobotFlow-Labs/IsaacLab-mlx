# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLX-compatible replacements for small Warp helper kernels."""

from __future__ import annotations

from typing import Any


def _mx_module():
    import mlx.core as mx

    return mx


def _as_mx_array(value: Any, *, dtype: Any | None = None):
    mx = _mx_module()
    if value is None:
        return None
    module_name = getattr(value.__class__, "__module__", "")
    array = value if module_name.startswith("mlx") else mx.array(value)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


def _is_present(value: Any) -> bool:
    return value is not None and getattr(value, "size", 0) != 0


def _quat_conjugate_xyzw(quat):
    mx = _mx_module()
    return mx.concatenate([-quat[..., :3], quat[..., 3:4]], axis=-1)


def _quat_rotate_xyzw(quat, vector):
    mx = _mx_module()
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    cross_1 = mx.linalg.cross(quat_xyz, vector, axis=-1)
    cross_2 = mx.linalg.cross(quat_xyz, cross_1, axis=-1)
    return vector + 2.0 * (quat_w * cross_1 + cross_2)


def _quat_rotate_inv_xyzw(quat, vector):
    return _quat_rotate_xyzw(_quat_conjugate_xyzw(quat), vector)


def _resolve_pair_indices(env_ids: Any, body_ids: Any, num_bodies: int):
    mx = _mx_module()
    env_ids_mx = _as_mx_array(env_ids, dtype=mx.int32)
    body_ids_mx = _as_mx_array(body_ids, dtype=mx.int32)
    env_grid = mx.broadcast_to(mx.expand_dims(env_ids_mx, axis=1), (env_ids_mx.shape[0], body_ids_mx.shape[0]))
    body_grid = mx.broadcast_to(mx.expand_dims(body_ids_mx, axis=0), (env_ids_mx.shape[0], body_ids_mx.shape[0]))
    flat_indices = env_grid.astype(mx.int32) * int(num_bodies) + body_grid.astype(mx.int32)
    return env_grid, body_grid, flat_indices


def _scatter_add(flattened, flat_indices, values):
    return flattened.at[flat_indices].add(values)


def _scatter_set(flattened, flat_indices, values):
    current = flattened[flat_indices]
    return flattened.at[flat_indices].add(values - current)


def reshape_tiled_image_mlx(
    tiled_image_buffer: Any,
    *,
    num_cameras: int,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_tiles_x: int,
):
    """Reshape a flattened tiled camera buffer to `(camera, height, width, channels)`."""

    mx = _mx_module()
    tiled_buffer = _as_mx_array(tiled_image_buffer)
    pixels_per_tiled_row = int(num_tiles_x) * int(image_width) * int(num_channels)
    if pixels_per_tiled_row <= 0:
        raise ValueError("Expected positive tiled image dimensions.")
    if int(tiled_buffer.size) % pixels_per_tiled_row != 0:
        raise ValueError("Tiled image buffer size is incompatible with the provided shape metadata.")

    tiled_height = int(tiled_buffer.size) // pixels_per_tiled_row
    if tiled_height % int(image_height) != 0:
        raise ValueError("Tiled image height is incompatible with the provided per-camera image height.")

    num_tiles_y = tiled_height // int(image_height)
    tiled_grid = mx.reshape(
        tiled_buffer,
        (num_tiles_y, int(image_height), int(num_tiles_x), int(image_width), int(num_channels)),
    )
    batched = mx.transpose(tiled_grid, (0, 2, 1, 3, 4))
    batched = mx.reshape(
        batched,
        (num_tiles_y * int(num_tiles_x), int(image_height), int(image_width), int(num_channels)),
    )
    return batched[: int(num_cameras)]


def add_forces_and_torques_at_position_mlx(
    env_ids: Any,
    body_ids: Any,
    forces: Any,
    torques: Any,
    positions: Any,
    link_positions: Any,
    link_quaternions: Any,
    composed_forces_b: Any,
    composed_torques_b: Any,
    *,
    is_global: bool,
):
    """MLX replacement for the Warp add helper used by `WrenchComposer`."""

    composed_forces = _as_mx_array(composed_forces_b)
    composed_torques = _as_mx_array(composed_torques_b)
    env_grid, body_grid, flat_indices = _resolve_pair_indices(env_ids, body_ids, composed_forces.shape[1])
    link_positions_mx = _as_mx_array(link_positions)
    link_quaternions_mx = _as_mx_array(link_quaternions)
    selected_positions = link_positions_mx[env_grid, body_grid]
    selected_quaternions = link_quaternions_mx[env_grid, body_grid]

    flat_forces = composed_forces.reshape((-1, 3))
    flat_torques = composed_torques.reshape((-1, 3))

    if _is_present(forces):
        forces_mx = _as_mx_array(forces, dtype=flat_forces.dtype)
        forces_link = _quat_rotate_inv_xyzw(selected_quaternions, forces_mx) if is_global else forces_mx
        flat_forces = _scatter_add(flat_forces, flat_indices, forces_link)
        if _is_present(positions):
            positions_mx = _as_mx_array(positions, dtype=flat_torques.dtype)
            positions_link = positions_mx - selected_positions if is_global else positions_mx
            torques_from_forces = _mx_module().linalg.cross(positions_link, forces_link, axis=-1)
            flat_torques = _scatter_add(flat_torques, flat_indices, torques_from_forces)

    if _is_present(torques):
        torques_mx = _as_mx_array(torques, dtype=flat_torques.dtype)
        torques_link = _quat_rotate_inv_xyzw(selected_quaternions, torques_mx) if is_global else torques_mx
        flat_torques = _scatter_add(flat_torques, flat_indices, torques_link)

    return flat_forces.reshape(composed_forces.shape), flat_torques.reshape(composed_torques.shape)


def set_forces_and_torques_at_position_mlx(
    env_ids: Any,
    body_ids: Any,
    forces: Any,
    torques: Any,
    positions: Any,
    link_positions: Any,
    link_quaternions: Any,
    composed_forces_b: Any,
    composed_torques_b: Any,
    *,
    is_global: bool,
):
    """MLX replacement for the Warp set helper used by `WrenchComposer`."""

    composed_forces = _as_mx_array(composed_forces_b)
    composed_torques = _as_mx_array(composed_torques_b)
    env_grid, body_grid, flat_indices = _resolve_pair_indices(env_ids, body_ids, composed_forces.shape[1])
    link_positions_mx = _as_mx_array(link_positions)
    link_quaternions_mx = _as_mx_array(link_quaternions)
    selected_positions = link_positions_mx[env_grid, body_grid]
    selected_quaternions = link_quaternions_mx[env_grid, body_grid]

    flat_forces = composed_forces.reshape((-1, 3))
    flat_torques = composed_torques.reshape((-1, 3))

    if _is_present(torques):
        torques_mx = _as_mx_array(torques, dtype=flat_torques.dtype)
        torques_link = _quat_rotate_inv_xyzw(selected_quaternions, torques_mx) if is_global else torques_mx
        flat_torques = _scatter_set(flat_torques, flat_indices, torques_link)

    if _is_present(forces):
        forces_mx = _as_mx_array(forces, dtype=flat_forces.dtype)
        forces_link = _quat_rotate_inv_xyzw(selected_quaternions, forces_mx) if is_global else forces_mx
        flat_forces = _scatter_set(flat_forces, flat_indices, forces_link)
        if _is_present(positions):
            positions_mx = _as_mx_array(positions, dtype=flat_torques.dtype)
            positions_link = positions_mx - selected_positions if is_global else positions_mx
            torques_from_forces = _mx_module().linalg.cross(positions_link, forces_link, axis=-1)
            flat_torques = _scatter_set(flat_torques, flat_indices, torques_from_forces)

    return flat_forces.reshape(composed_forces.shape), flat_torques.reshape(composed_torques.shape)


def detect_cpu_fallback(runtime_state: dict[str, Any]) -> dict[str, Any]:
    """Report whether benchmark runtime metadata indicates CPU fallback for kernel execution."""

    kernel_backend = runtime_state.get("kernel_backend")
    detected = kernel_backend == "cpu"
    return {
        "detected": detected,
        "reason": "kernel-backend=cpu" if detected else None,
        "active_kernel_backend": kernel_backend,
        "expected_kernel_backend": "metal",
    }
