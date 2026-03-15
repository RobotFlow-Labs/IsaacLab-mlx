# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable batched state primitives for mac-native simulator adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import mlx.core as mx


def _reshape_rows(values: Any, rows: int, columns: int, *, dtype=mx.float32) -> mx.array:
    """Normalize scalar/vector/matrix inputs into a `(rows, columns)` MLX array."""
    array = mx.array(values, dtype=dtype)
    if array.ndim == 0:
        return mx.full((rows, columns), array.item(), dtype=dtype)
    if array.ndim == 1:
        if array.shape[0] == columns:
            return mx.broadcast_to(array.reshape((1, columns)), (rows, columns))
        if columns == 1 and array.shape[0] == rows:
            return array.reshape((rows, 1))
        if array.shape[0] == rows * columns:
            return array.reshape((rows, columns))
    if array.ndim == 2 and array.shape == (1, columns):
        return mx.broadcast_to(array, (rows, columns))
    if array.ndim == 2 and array.shape == (rows, columns):
        return array
    return array.reshape((rows, columns))


def env_ids_to_array(env_ids: Sequence[int] | None, num_envs: int) -> mx.array:
    """Convert optional env ids into a stable MLX integer array."""
    if env_ids is None:
        env_ids = range(num_envs)
    return mx.array(list(env_ids), dtype=mx.int32)


class BatchedArticulationState:
    """Shared joint-state and effort-target buffers for batched articulated tasks."""

    def __init__(self, num_envs: int, num_joints: int):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.joint_pos = mx.zeros((num_envs, num_joints), dtype=mx.float32)
        self.joint_vel = mx.zeros((num_envs, num_joints), dtype=mx.float32)
        self.joint_effort_target = mx.zeros((num_envs, num_joints), dtype=mx.float32)

    def read(self) -> tuple[mx.array, mx.array]:
        return self.joint_pos, self.joint_vel

    def reset_envs(
        self,
        env_ids: Sequence[int],
        *,
        joint_pos: Any = 0.0,
        joint_vel: Any = 0.0,
        joint_effort_target: Any = 0.0,
    ) -> None:
        if len(env_ids) == 0:
            return
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(env_ids)
        self.joint_pos[ids] = _reshape_rows(joint_pos, rows, self.num_joints)
        self.joint_vel[ids] = _reshape_rows(joint_vel, rows, self.num_joints)
        self.joint_effort_target[ids] = _reshape_rows(joint_effort_target, rows, self.num_joints)

    def write(self, joint_pos: Any, joint_vel: Any, *, env_ids: Sequence[int] | None = None) -> None:
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(ids)
        self.joint_pos[ids] = _reshape_rows(joint_pos, rows, self.num_joints)
        self.joint_vel[ids] = _reshape_rows(joint_vel, rows, self.num_joints)

    def set_effort_target(self, efforts: Any, *, joint_ids: Sequence[int] | None = None) -> None:
        if joint_ids is None:
            joint_ids = tuple(range(self.num_joints))
        joint_ids = list(joint_ids)
        width = len(joint_ids)
        values = _reshape_rows(efforts, self.num_envs, width)
        self.joint_effort_target[:, joint_ids] = values


class EnvironmentOriginGrid:
    """Shared environment-origin helper for batched mac-sim tasks."""

    def __init__(self, num_envs: int, spacing: float):
        self.num_envs = num_envs
        self.spacing = spacing
        self.origins = self._build_origins(num_envs, spacing)

    @staticmethod
    def _build_origins(num_envs: int, spacing: float) -> mx.array:
        side = int(math.ceil(math.sqrt(num_envs)))
        grid_x = mx.arange(side, dtype=mx.float32)
        grid_y = mx.arange(side, dtype=mx.float32)
        mesh_x = mx.repeat(grid_x.reshape((1, -1)), side, axis=0).reshape((-1,))
        mesh_y = mx.repeat(grid_y.reshape((-1, 1)), side, axis=1).reshape((-1,))
        centered_x = (mesh_x[:num_envs] - (side - 1) / 2.0) * spacing
        centered_y = (mesh_y[:num_envs] - (side - 1) / 2.0) * spacing
        zeros = mx.zeros((num_envs,), dtype=mx.float32)
        return mx.stack([centered_x, centered_y, zeros], axis=-1)

    def positions_with_offset(self, env_ids: Sequence[int], offset: Any) -> mx.array:
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(ids)
        return self.origins[ids] + _reshape_rows(offset, rows, 3)


class BatchedRootState:
    """Shared root-state buffers for batched mac-native simulators."""

    def __init__(self, num_envs: int, *, origin_grid: EnvironmentOriginGrid | None = None):
        self.num_envs = num_envs
        self.origin_grid = origin_grid
        self.root_pos_w = mx.zeros((num_envs, 3), dtype=mx.float32)
        self.root_lin_vel_b = mx.zeros((num_envs, 3), dtype=mx.float32)
        self.root_ang_vel_b = mx.zeros((num_envs, 3), dtype=mx.float32)
        self.root_quat_w = mx.tile(mx.array([[0.0, 0.0, 0.0, 1.0]], dtype=mx.float32), (num_envs, 1))
        self.projected_gravity_b = mx.tile(mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32), (num_envs, 1))

    @property
    def env_origins(self) -> mx.array:
        if self.origin_grid is None:
            return mx.zeros((self.num_envs, 3), dtype=mx.float32)
        return self.origin_grid.origins

    def reset_envs(
        self,
        env_ids: Sequence[int],
        *,
        root_pos_w: Any = 0.0,
        root_quat_w: Any = (0.0, 0.0, 0.0, 1.0),
        root_lin_vel_b: Any = 0.0,
        root_ang_vel_b: Any = 0.0,
        projected_gravity_b: Any = (0.0, 0.0, -1.0),
    ) -> None:
        if len(env_ids) == 0:
            return
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(env_ids)
        self.root_pos_w[ids] = _reshape_rows(root_pos_w, rows, 3)
        self.root_quat_w[ids] = _reshape_rows(root_quat_w, rows, 4)
        self.root_lin_vel_b[ids] = _reshape_rows(root_lin_vel_b, rows, 3)
        self.root_ang_vel_b[ids] = _reshape_rows(root_ang_vel_b, rows, 3)
        self.projected_gravity_b[ids] = _reshape_rows(projected_gravity_b, rows, 3)

    def write_root_pose(self, root_pose: Any, *, env_ids: Sequence[int] | None = None) -> None:
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(ids)
        pose = mx.array(root_pose, dtype=mx.float32).reshape((rows, -1))
        self.root_pos_w[ids] = pose[:, :3]
        if pose.shape[1] >= 7:
            self.root_quat_w[ids] = pose[:, 3:7]

    def write_root_velocity(self, root_velocity: Any, *, env_ids: Sequence[int] | None = None) -> None:
        ids = env_ids_to_array(env_ids, self.num_envs)
        rows = len(ids)
        velocity = mx.array(root_velocity, dtype=mx.float32).reshape((rows, -1))
        self.root_lin_vel_b[ids] = velocity[:, :3]
        if velocity.shape[1] >= 6:
            self.root_ang_vel_b[ids] = velocity[:, 3:6]


@dataclass
class MacSimArticulationView:
    """Generic batched articulation view for the shared mac-sim scene substrate."""

    name: str
    joint_state: BatchedArticulationState
    root_state: BatchedRootState | None = None
    default_joint_pos: Any = 0.0
    default_joint_vel: Any = 0.0
    default_joint_effort_target: Any = 0.0
    default_root_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    default_root_velocity: Any = 0.0
    default_projected_gravity_b: Any = (0.0, 0.0, -1.0)

    @property
    def num_envs(self) -> int:
        return self.joint_state.num_envs

    @property
    def num_joints(self) -> int:
        return self.joint_state.num_joints

    def reset_envs(self, env_ids: Sequence[int] | None = None) -> None:
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if not ids:
            return
        self.joint_state.reset_envs(
            ids,
            joint_pos=self.default_joint_pos,
            joint_vel=self.default_joint_vel,
            joint_effort_target=self.default_joint_effort_target,
        )
        if self.root_state is not None:
            pose = mx.array(self.default_root_pose, dtype=mx.float32).reshape((1, -1))
            root_pos = pose[:, :3]
            root_quat = pose[:, 3:7] if pose.shape[1] >= 7 else (0.0, 0.0, 0.0, 1.0)
            self.root_state.reset_envs(
                ids,
                root_pos_w=root_pos,
                root_quat_w=root_quat,
                root_lin_vel_b=self.default_root_velocity,
                root_ang_vel_b=self.default_root_velocity,
                projected_gravity_b=self.default_projected_gravity_b,
            )

    def get_joint_state(self) -> tuple[mx.array, mx.array]:
        return self.joint_state.read()

    def set_joint_effort_target(self, efforts: Any, *, joint_ids: Sequence[int] | None = None) -> None:
        self.joint_state.set_effort_target(efforts, joint_ids=joint_ids)

    def write_joint_state(self, joint_pos: Any, joint_vel: Any, *, env_ids: Sequence[int] | None = None) -> None:
        self.joint_state.write(joint_pos, joint_vel, env_ids=env_ids)

    def write_root_pose(self, root_pose: Any, *, env_ids: Sequence[int] | None = None) -> None:
        if self.root_state is None:
            raise ValueError(f"Articulation '{self.name}' does not expose root-state IO.")
        self.root_state.write_root_pose(root_pose, env_ids=env_ids)

    def write_root_velocity(self, root_velocity: Any, *, env_ids: Sequence[int] | None = None) -> None:
        if self.root_state is None:
            raise ValueError(f"Articulation '{self.name}' does not expose root-state IO.")
        self.root_state.write_root_velocity(root_velocity, env_ids=env_ids)

    def state_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "num_envs": self.num_envs,
            "num_joints": self.num_joints,
            "joint_state_shape": list(self.joint_state.joint_pos.shape),
            "root_state_io": self.root_state is not None,
        }
        if self.root_state is not None:
            payload["root_state_shape"] = list(self.root_state.root_pos_w.shape)
        return payload


class MacSimSceneState:
    """Generic batched scene/articulation substrate for the shared mac-sim backend."""

    def __init__(self, num_envs: int, *, physics_dt: float = 1.0 / 60.0):
        self.num_envs = num_envs
        self.physics_dt = physics_dt
        self.articulations: dict[str, MacSimArticulationView] = {}
        self.step_count = 0
        self.reset_count = 0
        self.last_step_args: tuple[bool, bool] | None = None

    def add_articulation(
        self,
        name: str,
        *,
        num_joints: int,
        with_root_state: bool = False,
        origin_grid: EnvironmentOriginGrid | None = None,
        default_joint_pos: Any = 0.0,
        default_joint_vel: Any = 0.0,
        default_joint_effort_target: Any = 0.0,
        default_root_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        default_root_velocity: Any = 0.0,
    ) -> MacSimArticulationView:
        root_state = BatchedRootState(self.num_envs, origin_grid=origin_grid) if with_root_state else None
        articulation = MacSimArticulationView(
            name=name,
            joint_state=BatchedArticulationState(self.num_envs, num_joints=num_joints),
            root_state=root_state,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            default_joint_effort_target=default_joint_effort_target,
            default_root_pose=default_root_pose,
            default_root_velocity=default_root_velocity,
        )
        articulation.reset_envs()
        self.articulations[name] = articulation
        return articulation

    def get_articulation(self, name: str) -> MacSimArticulationView:
        try:
            return self.articulations[name]
        except KeyError as error:
            raise KeyError(f"Unknown mac-sim articulation '{name}'.") from error

    def reset(self, *, soft: bool = False) -> dict[str, Any]:
        self.reset_count += 1
        if not soft:
            self.step_count = 0
        for articulation in self.articulations.values():
            articulation.reset_envs()
        return self.state_dict()

    def step(self, *, render: bool = True, update_fabric: bool = False) -> dict[str, Any]:
        self.step_count += 1
        self.last_step_args = (render, update_fabric)
        return self.state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "physics_dt": self.physics_dt,
            "step_count": self.step_count,
            "reset_count": self.reset_count,
            "articulation_count": len(self.articulations),
            "articulations": {name: articulation.state_dict() for name, articulation in self.articulations.items()},
            "last_step_args": self.last_step_args,
        }
