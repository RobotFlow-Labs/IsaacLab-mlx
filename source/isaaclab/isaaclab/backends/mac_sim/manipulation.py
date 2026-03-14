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

from .env_cfgs import MacFrankaLiftEnvCfg, MacFrankaReachEnvCfg
from .hotpath import franka_end_effector_position_hotpath, franka_lift_object_step_hotpath
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
        step_reward = self.reward_buf
        step_terminated = self.reset_terminated
        step_time_outs = self.reset_time_outs

        reset_ids = [index for index, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            ids = mx.array(reset_ids, dtype=mx.int32)
            extras = {
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


def train_franka_lift_policy(cfg: MacFrankaLiftTrainCfg) -> dict[str, Any]:
    """Train a lightweight continuous-control Franka lift policy on the mac-native MLX slice."""

    mx.random.seed(cfg.env.seed)
    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    env = MacFrankaLiftEnv(cfg.env)
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
                f"[mlx-franka-lift] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id="Isaac-Lift-Cube-Franka-v0",
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


def play_franka_lift_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacFrankaLiftEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained Franka lift policy greedily and return episode returns."""

    cfg = env_cfg or MacFrankaLiftEnvCfg(num_envs=1)
    return play_gaussian_policy_checkpoint(
        checkpoint_path,
        env_factory=MacFrankaLiftEnv,
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
