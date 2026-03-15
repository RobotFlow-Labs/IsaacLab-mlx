# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native Franka manipulation slices for the MLX/mac-sim path."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from isaaclab.backends.runtime import (
    ArticulationCapabilities,
    MacSimBackend,
    SimBackendContract,
    SimCapabilities,
    resolve_runtime_selection,
    set_runtime_selection,
)
from isaaclab.utils.configclass import configclass

from .env_cfgs import (
    MacFrankaCabinetEnvCfg,
    MacFrankaLiftEnvCfg,
    MacFrankaOpenDrawerEnvCfg,
    MacFrankaReachEnvCfg,
    MacFrankaStackEnvCfg,
    MacFrankaStackInstanceRandomizeEnvCfg,
    MacFrankaStackRgbEnvCfg,
    MacFrankaTeddyBearLiftEnvCfg,
)
from .hotpath import (
    franka_cabinet_step_hotpath,
    franka_end_effector_position_hotpath,
    franka_lift_object_step_hotpath,
    franka_stack_object_step_hotpath,
    franka_stack_rgb_step_hotpath,
    get_franka_hotpath_backend,
)
from .ppo_training import (
    build_checkpoint_metadata,
    compute_gae,
    mean_recent_return,
    normalize_advantages,
    play_gaussian_policy_checkpoint,
    resolve_resume_hidden_dim,
    save_policy_checkpoint,
)
from .reset_primitives import DeterministicResetSampler
from .state_primitives import BatchedArticulationState

LOG_2_PI = math.log(2.0 * math.pi)


@configclass
class MacFrankaReachTrainCfg:
    """Training configuration for the MLX Franka reach slice."""

    env: MacFrankaReachEnvCfg = MacFrankaReachEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_reach_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaLiftTrainCfg:
    """Training configuration for the MLX Franka lift slice."""

    env: MacFrankaLiftEnvCfg = MacFrankaLiftEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_lift_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaTeddyBearLiftTrainCfg:
    """Training configuration for the MLX Franka teddy-bear lift slice."""

    env: MacFrankaTeddyBearLiftEnvCfg = MacFrankaTeddyBearLiftEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_teddy_bear_lift_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaStackTrainCfg:
    """Training configuration for the MLX Franka stack slice."""

    env: MacFrankaStackEnvCfg = MacFrankaStackEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_stack_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaStackInstanceRandomizeTrainCfg:
    """Training configuration for the MLX Franka instance-randomized stack slice."""

    env: MacFrankaStackInstanceRandomizeEnvCfg = MacFrankaStackInstanceRandomizeEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_stack_instance_randomize_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaCabinetTrainCfg:
    """Training configuration for the MLX Franka cabinet slice."""

    env: MacFrankaCabinetEnvCfg = MacFrankaCabinetEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_cabinet_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaOpenDrawerTrainCfg:
    """Training configuration for the MLX Franka open-drawer slice."""

    env: MacFrankaOpenDrawerEnvCfg = MacFrankaOpenDrawerEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_open_drawer_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None


@configclass
class MacFrankaStackRgbTrainCfg:
    """Training configuration for the MLX three-cube Franka stack slice."""

    env: MacFrankaStackRgbEnvCfg = MacFrankaStackRgbEnvCfg()
    hidden_dim: int = 128
    updates: int = 10
    rollout_steps: int = 24
    epochs_per_update: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    action_std: float = 0.25
    checkpoint_path: str = "logs/mlx/franka_stack_rgb_policy.npz"
    eval_interval: int = 5
    resume_from: str | None = None

class MacFrankaReachSimBackend(MacSimBackend):
    """A lightweight batched Franka reach backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=False,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )
    contract = SimBackendContract(
        reset_signature="reset(soft: bool = False) -> None",
        step_signature="step(render: bool = True, update_fabric: bool = False) -> None",
        articulations=ArticulationCapabilities(
            joint_state_io=True,
            root_state_io=False,
            effort_targets=True,
            batched_views=True,
        ),
    )

    def __init__(self, cfg: MacFrankaReachEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.reset_sampler = reset_sampler or DeterministicResetSampler(cfg.seed)
        self.state = BatchedArticulationState(cfg.num_envs, num_joints=cfg.action_space)
        self.default_joint_pos = mx.array(cfg.default_joint_pos, dtype=mx.float32).reshape((1, cfg.action_space))
        self.joint_lower_limits = mx.array(cfg.joint_lower_limits, dtype=mx.float32).reshape((1, cfg.action_space))
        self.joint_upper_limits = mx.array(cfg.joint_upper_limits, dtype=mx.float32).reshape((1, cfg.action_space))
        self.action_targets = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.joint_acc = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.applied_torque = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        self.target_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.ee_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.reset()

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    def reset(self, *, soft: bool = False) -> None:
        del soft
        self.reset_envs(list(range(self.num_envs)))

    def reset_envs(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        joint_pos = mx.broadcast_to(self.default_joint_pos, (rows, self.cfg.action_space)) + self.reset_sampler.uniform(
            (rows, self.cfg.action_space),
            -self.cfg.joint_reset_noise,
            self.cfg.joint_reset_noise,
        )
        joint_pos = mx.clip(
            joint_pos,
            mx.broadcast_to(self.joint_lower_limits, (rows, self.cfg.action_space)),
            mx.broadcast_to(self.joint_upper_limits, (rows, self.cfg.action_space)),
        )
        self.state.reset_envs(env_ids, joint_pos=joint_pos, joint_vel=0.0, joint_effort_target=0.0)
        self.action_targets[ids] = 0.0
        self.joint_acc[ids] = 0.0
        self.applied_torque[ids] = 0.0
        self.target_pos_w[ids, 0] = self.reset_sampler.uniform((rows,), self.cfg.target_x_range[0], self.cfg.target_x_range[1])
        self.target_pos_w[ids, 1] = self.reset_sampler.uniform((rows,), self.cfg.target_y_range[0], self.cfg.target_y_range[1])
        self.target_pos_w[ids, 2] = self.reset_sampler.uniform((rows,), self.cfg.target_z_range[0], self.cfg.target_z_range[1])
        self.ee_pos_w[ids] = franka_end_effector_position_hotpath(self.state.joint_pos[ids, :7])

    def set_action_targets(self, actions: Any) -> None:
        self.action_targets = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space)), -1.0, 1.0)

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        dt = self.physics_dt
        desired_joint_pos = self.default_joint_pos + self.cfg.action_scale * self.action_targets
        desired_joint_pos = mx.clip(
            desired_joint_pos,
            mx.broadcast_to(self.joint_lower_limits, desired_joint_pos.shape),
            mx.broadcast_to(self.joint_upper_limits, desired_joint_pos.shape),
        )
        pos_error = desired_joint_pos - self.state.joint_pos
        acc = self.cfg.joint_stiffness * pos_error - self.cfg.joint_damping * self.state.joint_vel
        acc = acc / self.cfg.joint_inertia
        self.state.joint_vel = self.state.joint_vel + dt * acc
        self.state.joint_pos = mx.clip(
            self.state.joint_pos + dt * self.state.joint_vel,
            mx.broadcast_to(self.joint_lower_limits, self.state.joint_pos.shape),
            mx.broadcast_to(self.joint_upper_limits, self.state.joint_pos.shape),
        )
        self.joint_acc = acc
        self.applied_torque = self.cfg.joint_stiffness * pos_error
        self.ee_pos_w = franka_end_effector_position_hotpath(self.state.joint_pos[:, :7])

    def goal_distance(self) -> mx.array:
        return mx.linalg.norm(self.target_pos_w - self.ee_pos_w, axis=1)

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        return self.state.read()

    def set_joint_effort_target(self, articulation: Any, efforts: Any, *, joint_ids: Any | None = None) -> None:
        del articulation
        self.state.set_effort_target(efforts, joint_ids=joint_ids)

    def write_joint_state(
        self,
        articulation: Any,
        joint_pos: Any,
        joint_vel: Any,
        *,
        joint_acc: Any | None = None,
        env_ids: Any | None = None,
    ) -> None:
        del articulation, joint_acc
        self.state.write(joint_pos, joint_vel, env_ids=env_ids)
        self.ee_pos_w = franka_end_effector_position_hotpath(self.state.joint_pos[:, :7])

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Any | None = None) -> None:
        del articulation, root_pose, env_ids

    def write_root_velocity(self, articulation: Any, root_velocity: Any, *, env_ids: Any | None = None) -> None:
        del articulation, root_velocity, env_ids

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "task": "franka-reach",
            "num_envs": self.num_envs,
            "capabilities": self.capabilities.__dict__,
            "contract": {
                "reset_signature": self.contract.reset_signature,
                "step_signature": self.contract.step_signature,
                "articulations": self.contract.articulations.__dict__,
            },
            "subsystems": {
                "analytic_kinematics": True,
                "object_tracking": False,
                "grasp_logic": False,
                "hotpath": get_franka_hotpath_backend(),
            },
            "joint_state_shape": list(self.state.joint_pos.shape),
        }


class MacFrankaLiftSimBackend(MacFrankaReachSimBackend):
    """A lightweight batched Franka cube-lift backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )

    def __init__(self, cfg: MacFrankaLiftEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.grasped = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        super().__init__(cfg, reset_sampler=reset_sampler)
        self.cfg = cfg

    def reset_envs(self, env_ids: list[int]) -> None:
        super().reset_envs(env_ids)
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        self.cube_pos_w[ids, 0] = self.reset_sampler.uniform((rows,), self.cfg.cube_x_range[0], self.cfg.cube_x_range[1])
        self.cube_pos_w[ids, 1] = self.reset_sampler.uniform((rows,), self.cfg.cube_y_range[0], self.cfg.cube_y_range[1])
        self.cube_pos_w[ids, 2] = self.cfg.table_height
        self.grasped[ids] = False

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        super().step(render=render, update_fabric=update_fabric)
        gripper_target, gripper_velocity, self.grasped, self.cube_pos_w = franka_lift_object_step_hotpath(
            self.cube_pos_w,
            self.ee_pos_w,
            self.state.joint_pos[:, 7],
            self.action_targets[:, 7],
            self.grasped,
            self.physics_dt,
            float(self.joint_lower_limits[0, 7].item()),
            float(self.joint_upper_limits[0, 7].item()),
            self.cfg.gripper_closed_threshold,
            self.cfg.grasp_distance_threshold,
            self.cfg.grasp_offset_z,
            self.cfg.table_height,
        )
        self.state.joint_vel[:, 7] = gripper_velocity
        self.state.joint_pos[:, 7] = gripper_target

    def lift_success(self) -> mx.array:
        return self.cube_pos_w[:, 2] >= self.cfg.lift_success_height

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-lift"
        payload["subsystems"] = {
            "analytic_kinematics": True,
            "object_tracking": True,
            "grasp_logic": True,
            "hotpath": get_franka_hotpath_backend(),
        }
        return payload


class MacFrankaTeddyBearLiftSimBackend(MacFrankaLiftSimBackend):
    """A lightweight batched Franka teddy-bear lift backend on MLX/mac-sim."""

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-teddy-bear-lift"
        payload["subsystems"] = {
            **payload["subsystems"],
            "manipulated_object": "teddy-bear",
        }
        return payload


class MacFrankaStackSimBackend(MacFrankaReachSimBackend):
    """A lightweight batched Franka cube-stack backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )

    def __init__(self, cfg: MacFrankaStackEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.support_cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.grasped = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        self.stacked = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        super().__init__(cfg, reset_sampler=reset_sampler)
        self.cfg = cfg

    def reset_envs(self, env_ids: list[int]) -> None:
        super().reset_envs(env_ids)
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        support_x = self.reset_sampler.uniform((rows,), self.cfg.support_cube_x_range[0], self.cfg.support_cube_x_range[1])
        support_y = self.reset_sampler.uniform((rows,), self.cfg.support_cube_y_range[0], self.cfg.support_cube_y_range[1])
        direction = mx.where(self.reset_sampler.uniform((rows,), 0.0, 1.0) > 0.5, 1.0, -1.0)
        offset_x = self.reset_sampler.uniform(
            (rows,),
            self.cfg.movable_cube_offset_x_range[0],
            self.cfg.movable_cube_offset_x_range[1],
        )
        offset_y = direction * self.reset_sampler.uniform(
            (rows,),
            self.cfg.movable_cube_offset_y_range[0],
            self.cfg.movable_cube_offset_y_range[1],
        )
        movable_x = mx.clip(support_x + offset_x, self.cfg.cube_x_range[0], self.cfg.cube_x_range[1])
        movable_y = mx.clip(support_y + offset_y, self.cfg.cube_y_range[0], self.cfg.cube_y_range[1])
        self.support_cube_pos_w[ids] = mx.stack(
            (
                support_x,
                support_y,
                mx.full((rows,), self.cfg.table_height, dtype=mx.float32),
            ),
            axis=-1,
        )
        self.cube_pos_w[ids] = mx.stack(
            (
                movable_x,
                movable_y,
                mx.full((rows,), self.cfg.table_height, dtype=mx.float32),
            ),
            axis=-1,
        )
        self.grasped[ids] = False
        self.stacked[ids] = False

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        MacFrankaReachSimBackend.step(self, render=render, update_fabric=update_fabric)
        gripper_target, gripper_velocity, self.grasped, self.stacked, self.cube_pos_w = franka_stack_object_step_hotpath(
            self.cube_pos_w,
            self.support_cube_pos_w,
            self.ee_pos_w,
            self.state.joint_pos[:, 7],
            self.action_targets[:, 7],
            self.grasped,
            self.stacked,
            self.physics_dt,
            float(self.joint_lower_limits[0, 7].item()),
            float(self.joint_upper_limits[0, 7].item()),
            self.cfg.gripper_closed_threshold,
            self.cfg.stack_release_open_threshold,
            self.cfg.grasp_distance_threshold,
            self.cfg.grasp_offset_z,
            self.cfg.table_height,
            self.cfg.stack_offset_z,
            self.cfg.stack_xy_threshold,
            self.cfg.stack_z_threshold,
        )
        self.state.joint_vel[:, 7] = gripper_velocity
        self.state.joint_pos[:, 7] = gripper_target

    def stack_target_pos_w(self) -> mx.array:
        return self.support_cube_pos_w + mx.array([0.0, 0.0, self.cfg.stack_offset_z], dtype=mx.float32)

    def stack_error(self) -> mx.array:
        return self.stack_target_pos_w() - self.cube_pos_w

    def stack_success(self) -> mx.array:
        return self.stacked

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-stack"
        payload["subsystems"] = {
            "analytic_kinematics": True,
            "object_tracking": True,
            "grasp_logic": True,
            "stack_logic": True,
            "hotpath": get_franka_hotpath_backend(),
        }
        return payload


class MacFrankaStackInstanceRandomizeSimBackend(MacFrankaStackSimBackend):
    """A lightweight batched Franka instance-randomized stack backend on MLX/mac-sim."""

    def __init__(
        self, cfg: MacFrankaStackInstanceRandomizeEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None
    ):
        self.support_variant_id = mx.zeros((cfg.num_envs,), dtype=mx.int32)
        self.movable_variant_id = mx.zeros((cfg.num_envs,), dtype=mx.int32)
        super().__init__(cfg, reset_sampler=reset_sampler)
        self.cfg = cfg

    def reset_envs(self, env_ids: list[int]) -> None:
        super().reset_envs(env_ids)
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        support_variant_id = self.reset_sampler.integers((rows,), 0, self.cfg.variant_count, dtype=mx.int32)
        movable_variant_id = self.reset_sampler.integers((rows,), 0, self.cfg.variant_count - 1, dtype=mx.int32)
        movable_variant_id = mx.where(movable_variant_id >= support_variant_id, movable_variant_id + 1, movable_variant_id)
        self.support_variant_id[ids] = support_variant_id
        self.movable_variant_id[ids] = movable_variant_id

    def variant_observations(self) -> mx.array:
        if self.cfg.variant_count <= 1:
            return mx.zeros((self.cfg.num_envs, 2), dtype=mx.float32)
        scale = 2.0 / float(self.cfg.variant_count - 1)
        support_variant = self.support_variant_id.astype(mx.float32) * scale - 1.0
        movable_variant = self.movable_variant_id.astype(mx.float32) * scale - 1.0
        return mx.stack((support_variant, movable_variant), axis=-1)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-stack-instance-randomize"
        payload["subsystems"] = {
            **payload.get("subsystems", {}),
            "instance_randomization": True,
            "variant_count": self.cfg.variant_count,
            "variant_labels": self.cfg.variant_labels,
        }
        return payload


class MacFrankaCabinetSimBackend(MacFrankaReachSimBackend):
    """A lightweight batched Franka cabinet-drawer backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )

    def __init__(self, cfg: MacFrankaCabinetEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.handle_anchor_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.handle_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.drawer_open_amount = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self.grasped_handle = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        self.drawer_opened = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        super().__init__(cfg, reset_sampler=reset_sampler)
        self.cfg = cfg

    def reset_envs(self, env_ids: list[int]) -> None:
        super().reset_envs(env_ids)
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        self.handle_anchor_pos_w[ids, 0] = self.reset_sampler.uniform(
            (rows,), self.cfg.handle_anchor_x_range[0], self.cfg.handle_anchor_x_range[1]
        )
        self.handle_anchor_pos_w[ids, 1] = self.reset_sampler.uniform(
            (rows,), self.cfg.handle_anchor_y_range[0], self.cfg.handle_anchor_y_range[1]
        )
        self.handle_anchor_pos_w[ids, 2] = self.reset_sampler.uniform(
            (rows,), self.cfg.handle_anchor_z_range[0], self.cfg.handle_anchor_z_range[1]
        )
        self.handle_pos_w[ids] = self.handle_anchor_pos_w[ids]
        self.drawer_open_amount[ids] = 0.0
        self.grasped_handle[ids] = False
        self.drawer_opened[ids] = False

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        MacFrankaReachSimBackend.step(self, render=render, update_fabric=update_fabric)
        (
            gripper_target,
            gripper_velocity,
            self.grasped_handle,
            self.drawer_opened,
            self.drawer_open_amount,
            self.handle_pos_w,
        ) = franka_cabinet_step_hotpath(
            self.handle_anchor_pos_w,
            self.ee_pos_w,
            self.state.joint_pos[:, 7],
            self.action_targets[:, 7],
            self.grasped_handle,
            self.drawer_opened,
            self.drawer_open_amount,
            self.physics_dt,
            float(self.joint_lower_limits[0, 7].item()),
            float(self.joint_upper_limits[0, 7].item()),
            self.cfg.gripper_closed_threshold,
            self.cfg.handle_grasp_threshold,
            self.cfg.drawer_open_distance_max,
            self.cfg.drawer_success_distance,
        )
        self.state.joint_vel[:, 7] = gripper_velocity
        self.state.joint_pos[:, 7] = gripper_target

    def handle_distance(self) -> mx.array:
        return mx.linalg.norm(self.handle_pos_w - self.ee_pos_w, axis=1)

    def drawer_open_ratio(self) -> mx.array:
        return mx.clip(self.drawer_open_amount / self.cfg.drawer_open_distance_max, 0.0, 1.0)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-cabinet"
        payload["subsystems"] = {
            "analytic_kinematics": True,
            "object_tracking": True,
            "grasp_logic": True,
            "drawer_logic": True,
            "hotpath": get_franka_hotpath_backend(),
        }
        return payload


class MacFrankaOpenDrawerSimBackend(MacFrankaCabinetSimBackend):
    """A lightweight batched Franka open-drawer backend on MLX/mac-sim."""

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-open-drawer"
        return payload


class MacFrankaStackRgbSimBackend(MacFrankaReachSimBackend):
    """A lightweight batched three-cube Franka stack backend on MLX/mac-sim."""

    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=False,
        planners=False,
    )

    def __init__(self, cfg: MacFrankaStackRgbEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.middle_cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.top_cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.support_cube_pos_w = mx.zeros((cfg.num_envs, 3), dtype=mx.float32)
        self.middle_grasped = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        self.top_grasped = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        self.middle_stacked = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        self.top_stacked = mx.zeros((cfg.num_envs,), dtype=mx.bool_)
        super().__init__(cfg, reset_sampler=reset_sampler)
        self.cfg = cfg

    def reset_envs(self, env_ids: list[int]) -> None:
        super().reset_envs(env_ids)
        if not env_ids:
            return
        ids = mx.array(env_ids, dtype=mx.int32)
        rows = len(env_ids)
        support_x = self.reset_sampler.uniform((rows,), self.cfg.support_cube_x_range[0], self.cfg.support_cube_x_range[1])
        support_y = self.reset_sampler.uniform((rows,), self.cfg.support_cube_y_range[0], self.cfg.support_cube_y_range[1])

        middle_direction = mx.where(self.reset_sampler.uniform((rows,), 0.0, 1.0) > 0.5, 1.0, -1.0)
        top_direction = -middle_direction
        middle_x = mx.clip(
            support_x
            + self.reset_sampler.uniform((rows,), self.cfg.middle_cube_offset_x_range[0], self.cfg.middle_cube_offset_x_range[1]),
            self.cfg.cube_x_range[0],
            self.cfg.cube_x_range[1],
        )
        middle_y = mx.clip(
            support_y
            + middle_direction
            * self.reset_sampler.uniform((rows,), self.cfg.middle_cube_offset_y_range[0], self.cfg.middle_cube_offset_y_range[1]),
            self.cfg.cube_y_range[0],
            self.cfg.cube_y_range[1],
        )
        top_x = mx.clip(
            support_x
            + self.reset_sampler.uniform((rows,), self.cfg.top_cube_offset_x_range[0], self.cfg.top_cube_offset_x_range[1]),
            self.cfg.cube_x_range[0],
            self.cfg.cube_x_range[1],
        )
        top_y = mx.clip(
            support_y
            + top_direction
            * self.reset_sampler.uniform((rows,), self.cfg.top_cube_offset_y_range[0], self.cfg.top_cube_offset_y_range[1]),
            self.cfg.cube_y_range[0],
            self.cfg.cube_y_range[1],
        )
        self.support_cube_pos_w[ids] = mx.stack(
            (support_x, support_y, mx.full((rows,), self.cfg.table_height, dtype=mx.float32)),
            axis=-1,
        )
        self.middle_cube_pos_w[ids] = mx.stack(
            (middle_x, middle_y, mx.full((rows,), self.cfg.table_height, dtype=mx.float32)),
            axis=-1,
        )
        self.top_cube_pos_w[ids] = mx.stack(
            (top_x, top_y, mx.full((rows,), self.cfg.table_height, dtype=mx.float32)),
            axis=-1,
        )
        self.middle_grasped[ids] = False
        self.top_grasped[ids] = False
        self.middle_stacked[ids] = False
        self.top_stacked[ids] = False

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        MacFrankaReachSimBackend.step(self, render=render, update_fabric=update_fabric)
        (
            gripper_target,
            gripper_velocity,
            self.middle_grasped,
            self.top_grasped,
            self.middle_stacked,
            self.top_stacked,
            self.middle_cube_pos_w,
            self.top_cube_pos_w,
        ) = franka_stack_rgb_step_hotpath(
            self.middle_cube_pos_w,
            self.top_cube_pos_w,
            self.support_cube_pos_w,
            self.ee_pos_w,
            self.state.joint_pos[:, 7],
            self.action_targets[:, 7],
            self.middle_grasped,
            self.top_grasped,
            self.middle_stacked,
            self.top_stacked,
            self.physics_dt,
            float(self.joint_lower_limits[0, 7].item()),
            float(self.joint_upper_limits[0, 7].item()),
            self.cfg.gripper_closed_threshold,
            self.cfg.stack_release_open_threshold,
            self.cfg.grasp_distance_threshold,
            self.cfg.grasp_offset_z,
            self.cfg.table_height,
            self.cfg.stack_offset_z,
            self.cfg.stack_xy_threshold,
            self.cfg.stack_z_threshold,
        )
        self.state.joint_vel[:, 7] = gripper_velocity
        self.state.joint_pos[:, 7] = gripper_target

    def active_is_top_cube(self) -> mx.array:
        return self.middle_stacked

    def active_cube_pos_w(self) -> mx.array:
        return mx.where(self.active_is_top_cube()[:, None], self.top_cube_pos_w, self.middle_cube_pos_w)

    def middle_stack_target_pos_w(self) -> mx.array:
        return self.support_cube_pos_w + mx.array([0.0, 0.0, self.cfg.stack_offset_z], dtype=mx.float32)

    def top_stack_target_pos_w(self) -> mx.array:
        return self.middle_cube_pos_w + mx.array([0.0, 0.0, self.cfg.stack_offset_z], dtype=mx.float32)

    def active_stack_target_pos_w(self) -> mx.array:
        return mx.where(self.active_is_top_cube()[:, None], self.top_stack_target_pos_w(), self.middle_stack_target_pos_w())

    def middle_stack_error(self) -> mx.array:
        return self.middle_stack_target_pos_w() - self.middle_cube_pos_w

    def top_stack_error(self) -> mx.array:
        return self.top_stack_target_pos_w() - self.top_cube_pos_w

    def active_stack_error(self) -> mx.array:
        return self.active_stack_target_pos_w() - self.active_cube_pos_w()

    def active_grasped(self) -> mx.array:
        return mx.where(self.active_is_top_cube(), self.top_grasped, self.middle_grasped)

    def stack_success(self) -> mx.array:
        return self.top_stacked

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["task"] = "franka-stack-rgb"
        payload["subsystems"] = {
            "analytic_kinematics": True,
            "object_tracking": True,
            "grasp_logic": True,
            "multi_object_logic": True,
            "stack_logic": True,
            "hotpath": get_franka_hotpath_backend(),
        }
        return payload


class MacFrankaReachEnv:
    """Vectorized Franka reach task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaReachEnvCfg | None = None):
        self.cfg = cfg or MacFrankaReachEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaReachSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def reset(self) -> tuple[dict[str, mx.array], dict[str, Any]]:
        self._reset_idx(list(range(self.num_envs)))
        self.obs_buf = self._get_observations()
        return self.obs_buf, {}

    def step(self, actions: Any) -> tuple[dict[str, mx.array], mx.array, mx.array, mx.array, dict[str, Any]]:
        self._pre_physics_step(actions)
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.sim_backend.step(render=False)

        self.episode_length_buf = self.episode_length_buf + 1
        self.reward_buf = self._get_rewards()
        self.episode_return_buf = self.episode_return_buf + self.reward_buf
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        # Snapshot per-step signals before any auto-reset mutates backend-backed buffers.
        step_reward = mx.array(self.reward_buf)
        step_terminated = mx.array(self.reset_terminated)
        step_time_outs = mx.array(self.reset_time_outs)

        reset_ids = [index for index, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            ids = mx.array(reset_ids, dtype=mx.int32)
            extras = {
                "reset_env_ids": reset_ids,
                "completed_lengths": [int(self.episode_length_buf[index].item()) for index in reset_ids],
                "completed_returns": [float(self.episode_return_buf[index].item()) for index in reset_ids],
                "terminated_env_ids": [index for index in reset_ids if bool(step_terminated[index].item())],
                "truncated_env_ids": [index for index in reset_ids if bool(step_time_outs[index].item())],
                "final_policy_observations": self._build_policy_observations()[ids],
            }
            self._reset_idx(reset_ids)

        self.obs_buf = self._get_observations()
        return self.obs_buf, step_reward, step_terminated, step_time_outs, extras

    def _pre_physics_step(self, actions: Any) -> None:
        self._actions = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space)), -1.0, 1.0)

    def _apply_action(self) -> None:
        self.sim_backend.set_action_targets(self._actions)

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        ee_error = self.sim_backend.target_pos_w - self.sim_backend.ee_pos_w
        return mx.concatenate((joint_pos, joint_vel, self.sim_backend.ee_pos_w, self.sim_backend.target_pos_w, ee_error), axis=-1)

    def _get_observations(self) -> dict[str, mx.array]:
        obs = self._build_policy_observations()
        self._previous_actions = self._actions
        return {"policy": obs}

    def _get_rewards(self) -> mx.array:
        distance = self.sim_backend.goal_distance()
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * distance)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        joint_vel_penalty = self.cfg.joint_vel_penalty_scale * mx.sum(mx.square(self.sim_backend.state.joint_vel), axis=1)
        success_bonus = self.cfg.success_bonus * (distance <= self.cfg.success_threshold).astype(mx.float32)
        return (reach_reward + action_penalty + joint_vel_penalty + success_bonus) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.goal_distance() <= self.cfg.success_threshold
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out

    def _reset_idx(self, env_ids: list[int]) -> None:
        if not env_ids:
            return
        self.sim_backend.reset_envs(env_ids)
        ids = mx.array(env_ids, dtype=mx.int32)
        self._actions[ids] = 0.0
        self._previous_actions[ids] = 0.0
        self.episode_return_buf[ids] = 0.0
        self.episode_length_buf[ids] = 0


class MacFrankaLiftEnv(MacFrankaReachEnv):
    """Vectorized Franka cube-lift task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaLiftEnvCfg | None = None):
        self.cfg = cfg or MacFrankaLiftEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaLiftSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        cube_error = self.sim_backend.cube_pos_w - self.sim_backend.ee_pos_w
        goal_gap = (self.cfg.lift_success_height - self.sim_backend.cube_pos_w[:, 2]).reshape((-1, 1))
        grasped = self.sim_backend.grasped.astype(mx.float32).reshape((-1, 1))
        return mx.concatenate(
            (joint_pos, joint_vel, self.sim_backend.ee_pos_w, self.sim_backend.cube_pos_w, cube_error, goal_gap, grasped),
            axis=-1,
        )

    def _get_rewards(self) -> mx.array:
        cube_error = mx.linalg.norm(self.sim_backend.cube_pos_w - self.sim_backend.ee_pos_w, axis=1)
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * cube_error)
        grasp_reward = self.cfg.grasp_reward_scale * self.sim_backend.grasped.astype(mx.float32)
        lift_reward = self.cfg.lift_reward_scale * mx.maximum(self.sim_backend.cube_pos_w[:, 2] - self.cfg.table_height, 0.0)
        success_bonus = self.cfg.lift_success_bonus * self.sim_backend.lift_success().astype(mx.float32)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        return (reach_reward + grasp_reward + lift_reward + success_bonus + action_penalty) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.lift_success()
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out


class MacFrankaTeddyBearLiftEnv(MacFrankaLiftEnv):
    """Vectorized Franka teddy-bear lift task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaTeddyBearLiftEnvCfg | None = None):
        self.cfg = cfg or MacFrankaTeddyBearLiftEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaTeddyBearLiftSimBackend(
            self.cfg,
            reset_sampler=self.reset_sampler.fork("sim-backend"),
        )
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()


class MacFrankaStackEnv(MacFrankaReachEnv):
    """Vectorized Franka cube-stack task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaStackEnvCfg | None = None):
        self.cfg = cfg or MacFrankaStackEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaStackSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        cube_error = self.sim_backend.cube_pos_w - self.sim_backend.ee_pos_w
        stack_error = self.sim_backend.stack_error()
        grasped = self.sim_backend.grasped.astype(mx.float32).reshape((-1, 1))
        stacked = self.sim_backend.stacked.astype(mx.float32).reshape((-1, 1))
        return mx.concatenate(
            (
                joint_pos,
                joint_vel,
                self.sim_backend.ee_pos_w,
                self.sim_backend.cube_pos_w,
                self.sim_backend.support_cube_pos_w,
                cube_error,
                stack_error,
                grasped,
                stacked,
            ),
            axis=-1,
        )

    def _get_rewards(self) -> mx.array:
        cube_error = mx.linalg.norm(self.sim_backend.cube_pos_w - self.sim_backend.ee_pos_w, axis=1)
        stack_error = mx.linalg.norm(self.sim_backend.stack_error(), axis=1)
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * cube_error)
        grasp_reward = self.cfg.grasp_reward_scale * self.sim_backend.grasped.astype(mx.float32)
        lift_reward = self.cfg.lift_reward_scale * mx.maximum(self.sim_backend.cube_pos_w[:, 2] - self.cfg.table_height, 0.0)
        stack_align_reward = self.cfg.stack_align_reward_scale * mx.exp(-self.cfg.stack_distance_reward_gain * stack_error)
        success_bonus = self.cfg.stack_success_bonus * self.sim_backend.stack_success().astype(mx.float32)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        joint_vel_penalty = self.cfg.joint_vel_penalty_scale * mx.sum(mx.square(self.sim_backend.state.joint_vel), axis=1)
        return (
            reach_reward
            + grasp_reward
            + lift_reward
            + stack_align_reward
            + success_bonus
            + action_penalty
            + joint_vel_penalty
        ) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.stack_success()
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out


class MacFrankaStackInstanceRandomizeEnv(MacFrankaStackEnv):
    """Vectorized Franka instance-randomized cube-stack task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaStackInstanceRandomizeEnvCfg | None = None):
        self.cfg = cfg or MacFrankaStackInstanceRandomizeEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaStackInstanceRandomizeSimBackend(
            self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend")
        )
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def _build_policy_observations(self) -> mx.array:
        base_obs = super()._build_policy_observations()
        return mx.concatenate((base_obs, self.sim_backend.variant_observations()), axis=-1)


class MacFrankaCabinetEnv(MacFrankaReachEnv):
    """Vectorized Franka cabinet-drawer task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaCabinetEnvCfg | None = None):
        self.cfg = cfg or MacFrankaCabinetEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaCabinetSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        handle_error = self.sim_backend.handle_pos_w - self.sim_backend.ee_pos_w
        drawer_open_amount = self.sim_backend.drawer_open_amount.reshape((-1, 1))
        grasped = self.sim_backend.grasped_handle.astype(mx.float32).reshape((-1, 1))
        opened = self.sim_backend.drawer_opened.astype(mx.float32).reshape((-1, 1))
        return mx.concatenate(
            (
                joint_pos,
                joint_vel,
                self.sim_backend.ee_pos_w,
                self.sim_backend.handle_pos_w,
                handle_error,
                drawer_open_amount,
                grasped,
                opened,
            ),
            axis=-1,
        )

    def _get_rewards(self) -> mx.array:
        handle_distance = self.sim_backend.handle_distance()
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * handle_distance)
        grasp_reward = self.cfg.grasp_reward_scale * self.sim_backend.grasped_handle.astype(mx.float32)
        open_reward = self.cfg.open_reward_scale * self.sim_backend.drawer_open_ratio()
        success_bonus = self.cfg.drawer_success_bonus * self.sim_backend.drawer_opened.astype(mx.float32)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        joint_vel_penalty = self.cfg.joint_vel_penalty_scale * mx.sum(mx.square(self.sim_backend.state.joint_vel), axis=1)
        return (reach_reward + grasp_reward + open_reward + success_bonus + action_penalty + joint_vel_penalty) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.drawer_opened
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out


class MacFrankaOpenDrawerEnv(MacFrankaCabinetEnv):
    """Vectorized Franka open-drawer task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaOpenDrawerEnvCfg | None = None):
        self.cfg = cfg or MacFrankaOpenDrawerEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaOpenDrawerSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self.reset()


class MacFrankaStackRgbEnv(MacFrankaReachEnv):
    """Vectorized three-cube Franka stack task for MLX/mac-sim."""

    def __init__(self, cfg: MacFrankaStackRgbEnvCfg | None = None):
        self.cfg = cfg or MacFrankaStackRgbEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.sim_backend = MacFrankaStackRgbSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self._actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self._previous_actions = mx.zeros((self.num_envs, self.cfg.action_space), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.obs_buf = {"policy": mx.zeros((self.num_envs, self.cfg.observation_space), dtype=mx.float32)}
        self._previous_middle_stacked = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset()

    def _pre_physics_step(self, actions: Any) -> None:
        self._previous_middle_stacked = mx.array(self.sim_backend.middle_stacked)
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: list[int]) -> None:
        super()._reset_idx(env_ids)
        if env_ids:
            ids = mx.array(env_ids, dtype=mx.int32)
            self._previous_middle_stacked[ids] = self.sim_backend.middle_stacked[ids]

    def _build_policy_observations(self) -> mx.array:
        joint_pos, joint_vel = self.sim_backend.get_joint_state(None)
        active_cube_error = self.sim_backend.active_cube_pos_w() - self.sim_backend.ee_pos_w
        middle_stack_error = self.sim_backend.middle_stack_error()
        top_stack_error = self.sim_backend.top_stack_error()
        return mx.concatenate(
            (
                joint_pos,
                joint_vel,
                self.sim_backend.ee_pos_w,
                self.sim_backend.middle_cube_pos_w,
                self.sim_backend.top_cube_pos_w,
                self.sim_backend.support_cube_pos_w,
                active_cube_error,
                middle_stack_error,
                top_stack_error,
                self.sim_backend.middle_grasped.astype(mx.float32).reshape((-1, 1)),
                self.sim_backend.top_grasped.astype(mx.float32).reshape((-1, 1)),
                self.sim_backend.middle_stacked.astype(mx.float32).reshape((-1, 1)),
                self.sim_backend.top_stacked.astype(mx.float32).reshape((-1, 1)),
                self.sim_backend.active_is_top_cube().astype(mx.float32).reshape((-1, 1)),
            ),
            axis=-1,
        )

    def _get_rewards(self) -> mx.array:
        active_cube_distance = mx.linalg.norm(self.sim_backend.active_cube_pos_w() - self.sim_backend.ee_pos_w, axis=1)
        active_stack_distance = mx.linalg.norm(self.sim_backend.active_stack_error(), axis=1)
        active_cube_height = self.sim_backend.active_cube_pos_w()[:, 2]
        reach_reward = self.cfg.reach_reward_scale * mx.exp(-self.cfg.distance_reward_gain * active_cube_distance)
        grasp_reward = self.cfg.grasp_reward_scale * self.sim_backend.active_grasped().astype(mx.float32)
        lift_reward = self.cfg.lift_reward_scale * mx.maximum(active_cube_height - self.cfg.table_height, 0.0)
        middle_stage_bonus = self.cfg.middle_stage_bonus * (
            self.sim_backend.middle_stacked & ~self._previous_middle_stacked
        ).astype(mx.float32)
        top_stack_align_reward = (
            self.cfg.top_stack_align_reward_scale
            * mx.exp(-self.cfg.top_stack_distance_reward_gain * active_stack_distance)
            * self.sim_backend.active_is_top_cube().astype(mx.float32)
        )
        success_bonus = self.cfg.stack_success_bonus * self.sim_backend.stack_success().astype(mx.float32)
        action_penalty = self.cfg.action_rate_penalty_scale * mx.sum(mx.square(self._actions - self._previous_actions), axis=1)
        joint_vel_penalty = self.cfg.joint_vel_penalty_scale * mx.sum(mx.square(self.sim_backend.state.joint_vel), axis=1)
        return (
            reach_reward
            + grasp_reward
            + lift_reward
            + middle_stage_bonus
            + top_stack_align_reward
            + success_bonus
            + action_penalty
            + joint_vel_penalty
        ) * self.step_dt

    def _get_dones(self) -> tuple[mx.array, mx.array]:
        success = self.sim_backend.stack_success()
        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out


class MacFrankaReachPolicy(nn.Module):
    """Small Gaussian PPO policy/value network for the Franka reach slice."""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.backbone = [
            nn.Linear(obs_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ]
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, obs: mx.array) -> tuple[mx.array, mx.array]:
        x = obs
        for layer in self.backbone:
            x = nn.elu(layer(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def _gaussian_log_probs(actions: mx.array, mean: mx.array, std: float) -> mx.array:
    variance = std * std
    return -0.5 * mx.sum(mx.square(actions - mean) / variance + math.log(variance) + LOG_2_PI, axis=-1)


def _gaussian_entropy(action_dim: int, std: float) -> float:
    return action_dim * (0.5 + 0.5 * LOG_2_PI + math.log(std))


def _ppo_loss(
    model: MacFrankaReachPolicy,
    obs: mx.array,
    actions: mx.array,
    old_log_probs: mx.array,
    advantages: mx.array,
    returns: mx.array,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    action_std: float,
) -> mx.array:
    mean, values = model(obs)
    log_probs = _gaussian_log_probs(actions, mean, action_std)
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -mx.mean(mx.minimum(ratio * advantages, clipped_ratio * advantages))
    value_loss = 0.5 * mx.mean(mx.square(returns - values))
    entropy_bonus = _gaussian_entropy(actions.shape[1], action_std)
    return policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus


def train_franka_reach_policy(cfg: MacFrankaReachTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka reach policy on the mac-native MLX slice."""

    mx.random.seed(cfg.env.seed)
    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    env = MacFrankaReachEnv(cfg.env)
    model = MacFrankaReachPolicy(
        obs_dim=cfg.env.observation_space,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.env.action_space,
    )
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
    resumed_from = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from does not exist: {resume_path}")
        model.load_weights(str(resume_path))
        resumed_from = str(resume_path)
    mx.eval(model.parameters())

    obs = env.reset()[0]["policy"]
    completed_returns: list[float] = []

    for update in range(cfg.updates):
        obs_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        rewards_rollout = []
        terminated_rollout = []
        values_rollout = []
        bootstrap_obs_rollout = []

        for _ in range(cfg.rollout_steps):
            mean, values = model(obs)
            noise = mx.random.normal(shape=mean.shape).astype(mx.float32)
            actions = mx.clip(mean + cfg.action_std * noise, -1.0, 1.0)
            next_obs, reward, terminated, _, extras = env.step(actions)
            bootstrap_obs = next_obs["policy"]
            if extras.get("reset_env_ids"):
                ids = mx.array(extras["reset_env_ids"], dtype=mx.int32)
                bootstrap_obs = bootstrap_obs + 0.0
                bootstrap_obs[ids] = extras["final_policy_observations"]

            obs_rollout.append(obs)
            actions_rollout.append(actions)
            log_probs_rollout.append(_gaussian_log_probs(actions, mean, cfg.action_std))
            rewards_rollout.append(reward)
            terminated_rollout.append(terminated.astype(mx.float32))
            values_rollout.append(values)
            bootstrap_obs_rollout.append(bootstrap_obs)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        rewards_t = mx.stack(rewards_rollout)
        terminated_t = mx.stack(terminated_rollout)
        values_t = mx.stack(values_rollout)
        flat_bootstrap_obs = mx.reshape(mx.stack(bootstrap_obs_rollout), (-1, cfg.env.observation_space))
        _, flat_next_values = model(flat_bootstrap_obs)
        next_values_t = mx.reshape(flat_next_values, (cfg.rollout_steps, cfg.env.num_envs))
        advantages, returns = compute_gae(
            rewards_t,
            terminated_t,
            values_t,
            next_values_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1, cfg.env.action_space))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))
        flat_advantages = normalize_advantages(flat_advantages)

        for _ in range(cfg.epochs_per_update):
            loss, grads = loss_and_grad(
                model,
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_advantages,
                flat_returns,
                cfg.clip_epsilon,
                cfg.value_loss_coef,
                cfg.entropy_coef,
                cfg.action_std,
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = float(mx.mean(rewards_t).item())
            mean_return = mean_recent_return(completed_returns)
            print(
                f"[mlx-franka-reach] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id="Isaac-Reach-Franka-v0",
            policy_distribution="gaussian",
            action_std=cfg.action_std,
            train_cfg=asdict(cfg),
        ),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "resumed_from": resumed_from,
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": mean_recent_return(completed_returns),
    }


def play_franka_reach_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaReachEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka reach policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaReachEnvCfg(num_envs=1)
    return play_gaussian_policy_checkpoint(
        checkpoint_path,
        env_factory=MacFrankaReachEnv,
        env_cfg=cfg,
        model_factory=lambda obs_dim, policy_hidden_dim, action_dim: MacFrankaReachPolicy(
            obs_dim=obs_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
        default_hidden_dim=128,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def _train_franka_lift_like_policy(
    cfg: MacFrankaLiftTrainCfg | MacFrankaTeddyBearLiftTrainCfg,
    *,
    env_factory: type[MacFrankaLiftEnv] | type[MacFrankaTeddyBearLiftEnv],
    task_id: str,
    log_prefix: str,
) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka lift-like policy on the mac-native MLX slice."""

    mx.random.seed(cfg.env.seed)
    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    env = env_factory(cfg.env)
    model = MacFrankaReachPolicy(
        obs_dim=cfg.env.observation_space,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.env.action_space,
    )
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
    resumed_from = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from does not exist: {resume_path}")
        model.load_weights(str(resume_path))
        resumed_from = str(resume_path)
    mx.eval(model.parameters())

    obs = env.reset()[0]["policy"]
    completed_returns: list[float] = []

    for update in range(cfg.updates):
        obs_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        rewards_rollout = []
        terminated_rollout = []
        values_rollout = []
        bootstrap_obs_rollout = []

        for _ in range(cfg.rollout_steps):
            mean, values = model(obs)
            noise = mx.random.normal(shape=mean.shape).astype(mx.float32)
            actions = mx.clip(mean + cfg.action_std * noise, -1.0, 1.0)
            next_obs, reward, terminated, _, extras = env.step(actions)
            bootstrap_obs = next_obs["policy"]
            if extras.get("reset_env_ids"):
                ids = mx.array(extras["reset_env_ids"], dtype=mx.int32)
                bootstrap_obs = bootstrap_obs + 0.0
                bootstrap_obs[ids] = extras["final_policy_observations"]

            obs_rollout.append(obs)
            actions_rollout.append(actions)
            log_probs_rollout.append(_gaussian_log_probs(actions, mean, cfg.action_std))
            rewards_rollout.append(reward)
            terminated_rollout.append(terminated.astype(mx.float32))
            values_rollout.append(values)
            bootstrap_obs_rollout.append(bootstrap_obs)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        rewards_t = mx.stack(rewards_rollout)
        terminated_t = mx.stack(terminated_rollout)
        values_t = mx.stack(values_rollout)
        flat_bootstrap_obs = mx.reshape(mx.stack(bootstrap_obs_rollout), (-1, cfg.env.observation_space))
        _, flat_next_values = model(flat_bootstrap_obs)
        next_values_t = mx.reshape(flat_next_values, (cfg.rollout_steps, cfg.env.num_envs))
        advantages, returns = compute_gae(
            rewards_t,
            terminated_t,
            values_t,
            next_values_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1, cfg.env.action_space))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))
        flat_advantages = normalize_advantages(flat_advantages)

        for _ in range(cfg.epochs_per_update):
            loss, grads = loss_and_grad(
                model,
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_advantages,
                flat_returns,
                cfg.clip_epsilon,
                cfg.value_loss_coef,
                cfg.entropy_coef,
                cfg.action_std,
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = float(mx.mean(rewards_t).item())
            mean_return = mean_recent_return(completed_returns)
            print(
                f"[{log_prefix}] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id=task_id,
            policy_distribution="gaussian",
            action_std=cfg.action_std,
            train_cfg=asdict(cfg),
        ),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "resumed_from": resumed_from,
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": mean_recent_return(completed_returns),
    }


def train_franka_lift_policy(cfg: MacFrankaLiftTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka lift policy on the mac-native MLX slice."""

    return _train_franka_lift_like_policy(
        cfg,
        env_factory=MacFrankaLiftEnv,
        task_id="Isaac-Lift-Cube-Franka-v0",
        log_prefix="mlx-franka-lift",
    )


def train_franka_teddy_bear_lift_policy(cfg: MacFrankaTeddyBearLiftTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka teddy-bear lift policy on the mac-native MLX slice."""

    return _train_franka_lift_like_policy(
        cfg,
        env_factory=MacFrankaTeddyBearLiftEnv,
        task_id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
        log_prefix="mlx-franka-teddy-bear-lift",
    )


def _play_franka_lift_like_policy(
    checkpoint_path: str,
    *,
    env_factory: type[MacFrankaLiftEnv] | type[MacFrankaTeddyBearLiftEnv],
    env_cfg: MacFrankaLiftEnvCfg | MacFrankaTeddyBearLiftEnvCfg,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka lift-like policy greedily and return episode returns."""

    return play_gaussian_policy_checkpoint(
        checkpoint_path,
        env_factory=env_factory,
        env_cfg=env_cfg,
        model_factory=lambda obs_dim, policy_hidden_dim, action_dim: MacFrankaReachPolicy(
            obs_dim=obs_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
        default_hidden_dim=128,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def play_franka_lift_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaLiftEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka lift policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaLiftEnvCfg(num_envs=1)
    return _play_franka_lift_like_policy(
        checkpoint_path,
        env_factory=MacFrankaLiftEnv,
        env_cfg=cfg,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def play_franka_teddy_bear_lift_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaTeddyBearLiftEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka teddy-bear lift policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaTeddyBearLiftEnvCfg(num_envs=1)
    return _play_franka_lift_like_policy(
        checkpoint_path,
        env_factory=MacFrankaTeddyBearLiftEnv,
        env_cfg=cfg,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def _train_franka_stack_like_policy(
    cfg: MacFrankaStackTrainCfg | MacFrankaStackInstanceRandomizeTrainCfg,
    *,
    env_factory: type[MacFrankaStackEnv] | type[MacFrankaStackInstanceRandomizeEnv],
    task_id: str,
    log_prefix: str,
) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka stack-like policy on the mac-native MLX slice."""

    mx.random.seed(cfg.env.seed)
    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    env = env_factory(cfg.env)
    model = MacFrankaReachPolicy(
        obs_dim=cfg.env.observation_space,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.env.action_space,
    )
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
    resumed_from = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from does not exist: {resume_path}")
        model.load_weights(str(resume_path))
        resumed_from = str(resume_path)
    mx.eval(model.parameters())

    obs = env.reset()[0]["policy"]
    completed_returns: list[float] = []

    for update in range(cfg.updates):
        obs_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        rewards_rollout = []
        terminated_rollout = []
        values_rollout = []
        bootstrap_obs_rollout = []

        for _ in range(cfg.rollout_steps):
            mean, values = model(obs)
            noise = mx.random.normal(shape=mean.shape).astype(mx.float32)
            actions = mx.clip(mean + cfg.action_std * noise, -1.0, 1.0)
            next_obs, reward, terminated, _, extras = env.step(actions)
            bootstrap_obs = next_obs["policy"]
            if extras.get("reset_env_ids"):
                ids = mx.array(extras["reset_env_ids"], dtype=mx.int32)
                bootstrap_obs = bootstrap_obs + 0.0
                bootstrap_obs[ids] = extras["final_policy_observations"]

            obs_rollout.append(obs)
            actions_rollout.append(actions)
            log_probs_rollout.append(_gaussian_log_probs(actions, mean, cfg.action_std))
            rewards_rollout.append(reward)
            terminated_rollout.append(terminated.astype(mx.float32))
            values_rollout.append(values)
            bootstrap_obs_rollout.append(bootstrap_obs)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        rewards_t = mx.stack(rewards_rollout)
        terminated_t = mx.stack(terminated_rollout)
        values_t = mx.stack(values_rollout)
        flat_bootstrap_obs = mx.reshape(mx.stack(bootstrap_obs_rollout), (-1, cfg.env.observation_space))
        _, flat_next_values = model(flat_bootstrap_obs)
        next_values_t = mx.reshape(flat_next_values, (cfg.rollout_steps, cfg.env.num_envs))
        advantages, returns = compute_gae(
            rewards_t,
            terminated_t,
            values_t,
            next_values_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1, cfg.env.action_space))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))
        flat_advantages = normalize_advantages(flat_advantages)

        for _ in range(cfg.epochs_per_update):
            loss, grads = loss_and_grad(
                model,
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_advantages,
                flat_returns,
                cfg.clip_epsilon,
                cfg.value_loss_coef,
                cfg.entropy_coef,
                cfg.action_std,
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = float(mx.mean(rewards_t).item())
            mean_return = mean_recent_return(completed_returns)
            print(
                f"[{log_prefix}] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id=task_id,
            policy_distribution="gaussian",
            action_std=cfg.action_std,
            train_cfg=asdict(cfg),
        ),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "resumed_from": resumed_from,
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": mean_recent_return(completed_returns),
    }


def train_franka_stack_policy(cfg: MacFrankaStackTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka stack policy on the mac-native MLX slice."""

    return _train_franka_stack_like_policy(
        cfg,
        env_factory=MacFrankaStackEnv,
        task_id="Isaac-Stack-Cube-Franka-v0",
        log_prefix="mlx-franka-stack",
    )


def train_franka_stack_instance_randomize_policy(cfg: MacFrankaStackInstanceRandomizeTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka instance-randomized stack policy on the mac-native MLX slice."""

    return _train_franka_stack_like_policy(
        cfg,
        env_factory=MacFrankaStackInstanceRandomizeEnv,
        task_id="Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
        log_prefix="mlx-franka-stack-instance-randomize",
    )


def _play_franka_stack_like_policy(
    checkpoint_path: str,
    *,
    env_factory: type[MacFrankaStackEnv] | type[MacFrankaStackInstanceRandomizeEnv],
    env_cfg: MacFrankaStackEnvCfg | MacFrankaStackInstanceRandomizeEnvCfg,
    episodes: int,
    hidden_dim: int | None,
) -> list[float]:
    """Run a trained Franka stack-like policy greedily and return episode returns."""

    return play_gaussian_policy_checkpoint(
        checkpoint_path,
        env_factory=env_factory,
        env_cfg=env_cfg,
        model_factory=lambda obs_dim, policy_hidden_dim, action_dim: MacFrankaReachPolicy(
            obs_dim=obs_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
        default_hidden_dim=128,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def play_franka_stack_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaStackEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka stack policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaStackEnvCfg(num_envs=1)
    return _play_franka_stack_like_policy(
        checkpoint_path,
        env_factory=MacFrankaStackEnv,
        env_cfg=cfg,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def play_franka_stack_instance_randomize_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaStackInstanceRandomizeEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka instance-randomized stack policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaStackInstanceRandomizeEnvCfg(num_envs=1)
    return _play_franka_stack_like_policy(
        checkpoint_path,
        env_factory=MacFrankaStackInstanceRandomizeEnv,
        env_cfg=cfg,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def _train_franka_drawer_policy(
    cfg: MacFrankaCabinetTrainCfg | MacFrankaOpenDrawerTrainCfg,
    *,
    env_factory: type[MacFrankaCabinetEnv] | type[MacFrankaOpenDrawerEnv],
    task_id: str,
    log_prefix: str,
) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka drawer policy on the mac-native MLX slice."""

    mx.random.seed(cfg.env.seed)
    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    env = env_factory(cfg.env)
    model = MacFrankaReachPolicy(
        obs_dim=cfg.env.observation_space,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.env.action_space,
    )
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
    resumed_from = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from does not exist: {resume_path}")
        model.load_weights(str(resume_path))
        resumed_from = str(resume_path)
    mx.eval(model.parameters())

    obs = env.reset()[0]["policy"]
    completed_returns: list[float] = []

    for update in range(cfg.updates):
        obs_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        rewards_rollout = []
        terminated_rollout = []
        values_rollout = []
        bootstrap_obs_rollout = []

        for _ in range(cfg.rollout_steps):
            mean, values = model(obs)
            noise = mx.random.normal(shape=mean.shape).astype(mx.float32)
            actions = mx.clip(mean + cfg.action_std * noise, -1.0, 1.0)
            next_obs, reward, terminated, _, extras = env.step(actions)
            bootstrap_obs = next_obs["policy"]
            if extras.get("reset_env_ids"):
                ids = mx.array(extras["reset_env_ids"], dtype=mx.int32)
                bootstrap_obs = bootstrap_obs + 0.0
                bootstrap_obs[ids] = extras["final_policy_observations"]

            obs_rollout.append(obs)
            actions_rollout.append(actions)
            log_probs_rollout.append(_gaussian_log_probs(actions, mean, cfg.action_std))
            rewards_rollout.append(reward)
            terminated_rollout.append(terminated.astype(mx.float32))
            values_rollout.append(values)
            bootstrap_obs_rollout.append(bootstrap_obs)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        rewards_t = mx.stack(rewards_rollout)
        terminated_t = mx.stack(terminated_rollout)
        values_t = mx.stack(values_rollout)
        flat_bootstrap_obs = mx.reshape(mx.stack(bootstrap_obs_rollout), (-1, cfg.env.observation_space))
        _, flat_next_values = model(flat_bootstrap_obs)
        next_values_t = mx.reshape(flat_next_values, (cfg.rollout_steps, cfg.env.num_envs))
        advantages, returns = compute_gae(
            rewards_t,
            terminated_t,
            values_t,
            next_values_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1, cfg.env.action_space))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))
        flat_advantages = normalize_advantages(flat_advantages)

        for _ in range(cfg.epochs_per_update):
            loss, grads = loss_and_grad(
                model,
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_advantages,
                flat_returns,
                cfg.clip_epsilon,
                cfg.value_loss_coef,
                cfg.entropy_coef,
                cfg.action_std,
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = float(mx.mean(rewards_t).item())
            mean_return = mean_recent_return(completed_returns)
            print(f"[{log_prefix}] update={update + 1}/{cfg.updates} mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}")

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id=task_id,
            policy_distribution="gaussian",
            action_std=cfg.action_std,
            train_cfg=asdict(cfg),
        ),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "resumed_from": resumed_from,
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": mean_recent_return(completed_returns),
    }


def train_franka_cabinet_policy(cfg: MacFrankaCabinetTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka cabinet policy on the mac-native MLX slice."""

    return _train_franka_drawer_policy(
        cfg,
        env_factory=MacFrankaCabinetEnv,
        task_id="Isaac-Franka-Cabinet-Direct-v0",
        log_prefix="mlx-franka-cabinet",
    )


def train_franka_open_drawer_policy(cfg: MacFrankaOpenDrawerTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka open-drawer policy on the mac-native MLX slice."""

    return _train_franka_drawer_policy(
        cfg,
        env_factory=MacFrankaOpenDrawerEnv,
        task_id="Isaac-Open-Drawer-Franka-v0",
        log_prefix="mlx-franka-open-drawer",
    )


def _play_franka_drawer_policy(
    checkpoint_path: str,
    *,
    env_factory: type[MacFrankaCabinetEnv] | type[MacFrankaOpenDrawerEnv],
    env_cfg: MacFrankaCabinetEnvCfg | MacFrankaOpenDrawerEnvCfg,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka drawer policy greedily and return episode returns."""

    return play_gaussian_policy_checkpoint(
        checkpoint_path,
        env_factory=env_factory,
        env_cfg=env_cfg,
        model_factory=lambda obs_dim, policy_hidden_dim, action_dim: MacFrankaReachPolicy(
            obs_dim=obs_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
        default_hidden_dim=128,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def play_franka_cabinet_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaCabinetEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka cabinet policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaCabinetEnvCfg(num_envs=1)
    return _play_franka_drawer_policy(
        checkpoint_path,
        env_factory=MacFrankaCabinetEnv,
        env_cfg=cfg,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def play_franka_open_drawer_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaOpenDrawerEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka open-drawer policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaOpenDrawerEnvCfg(num_envs=1)
    return _play_franka_drawer_policy(
        checkpoint_path,
        env_factory=MacFrankaOpenDrawerEnv,
        env_cfg=cfg,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )


def train_franka_stack_rgb_policy(cfg: MacFrankaStackRgbTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control three-cube Franka stack policy on the mac-native MLX slice."""

    mx.random.seed(cfg.env.seed)
    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    env = MacFrankaStackRgbEnv(cfg.env)
    model = MacFrankaReachPolicy(
        obs_dim=cfg.env.observation_space,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.env.action_space,
    )
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
    resumed_from = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from does not exist: {resume_path}")
        model.load_weights(str(resume_path))
        resumed_from = str(resume_path)
    mx.eval(model.parameters())

    obs = env.reset()[0]["policy"]
    completed_returns: list[float] = []

    for update in range(cfg.updates):
        obs_rollout = []
        actions_rollout = []
        log_probs_rollout = []
        rewards_rollout = []
        terminated_rollout = []
        values_rollout = []
        bootstrap_obs_rollout = []

        for _ in range(cfg.rollout_steps):
            mean, values = model(obs)
            noise = mx.random.normal(shape=mean.shape).astype(mx.float32)
            actions = mx.clip(mean + cfg.action_std * noise, -1.0, 1.0)
            next_obs, reward, terminated, _, extras = env.step(actions)
            bootstrap_obs = next_obs["policy"]
            if extras.get("reset_env_ids"):
                ids = mx.array(extras["reset_env_ids"], dtype=mx.int32)
                bootstrap_obs = bootstrap_obs + 0.0
                bootstrap_obs[ids] = extras["final_policy_observations"]

            obs_rollout.append(obs)
            actions_rollout.append(actions)
            log_probs_rollout.append(_gaussian_log_probs(actions, mean, cfg.action_std))
            rewards_rollout.append(reward)
            terminated_rollout.append(terminated.astype(mx.float32))
            values_rollout.append(values)
            bootstrap_obs_rollout.append(bootstrap_obs)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        rewards_t = mx.stack(rewards_rollout)
        terminated_t = mx.stack(terminated_rollout)
        values_t = mx.stack(values_rollout)
        flat_bootstrap_obs = mx.reshape(mx.stack(bootstrap_obs_rollout), (-1, cfg.env.observation_space))
        _, flat_next_values = model(flat_bootstrap_obs)
        next_values_t = mx.reshape(flat_next_values, (cfg.rollout_steps, cfg.env.num_envs))
        advantages, returns = compute_gae(
            rewards_t,
            terminated_t,
            values_t,
            next_values_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1, cfg.env.action_space))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))
        flat_advantages = normalize_advantages(flat_advantages)

        for _ in range(cfg.epochs_per_update):
            loss, grads = loss_and_grad(
                model,
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_advantages,
                flat_returns,
                cfg.clip_epsilon,
                cfg.value_loss_coef,
                cfg.entropy_coef,
                cfg.action_std,
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = float(mx.mean(rewards_t).item())
            mean_return = mean_recent_return(completed_returns)
            print(
                f"[mlx-franka-stack-rgb] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id="Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0",
            policy_distribution="gaussian",
            action_std=cfg.action_std,
            train_cfg=asdict(cfg),
        ),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "resumed_from": resumed_from,
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": mean_recent_return(completed_returns),
    }


def play_franka_stack_rgb_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaStackRgbEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained three-cube Franka stack policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaStackRgbEnvCfg(num_envs=1)
    return play_gaussian_policy_checkpoint(
        checkpoint_path,
        env_factory=MacFrankaStackRgbEnv,
        env_cfg=cfg,
        model_factory=lambda obs_dim, policy_hidden_dim, action_dim: MacFrankaReachPolicy(
            obs_dim=obs_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
        default_hidden_dim=128,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )
