# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native cartpole simulator and MLX training helpers."""

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

from .env_cfgs import MacCartpoleEnvCfg
from .ppo_training import (
    build_checkpoint_metadata,
    compute_gae,
    mean_recent_return,
    normalize_advantages,
    play_categorical_policy_checkpoint,
    read_checkpoint_metadata,
    resolve_resume_hidden_dim,
    save_policy_checkpoint,
)
from .reset_primitives import DeterministicResetSampler
from .state_primitives import BatchedArticulationState


@configclass
class MacCartpoleTrainCfg:
    """Training configuration for the MLX cartpole PPO baseline."""

    env: MacCartpoleEnvCfg = MacCartpoleEnvCfg()
    hidden_dim: int = 128
    updates: int = 200
    rollout_steps: int = 64
    epochs_per_update: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    checkpoint_path: str = "logs/mlx/cartpole_policy.npz"
    eval_interval: int = 10
    resume_from: str | None = None


class MacCartpoleSimBackend(MacSimBackend):
    """A batched cartpole simulator backed by MLX arrays."""

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

    def __init__(self, cfg: MacCartpoleEnvCfg, *, reset_sampler: DeterministicResetSampler | None = None):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.reset_sampler = reset_sampler or DeterministicResetSampler(cfg.seed)
        self.state = BatchedArticulationState(cfg.num_envs, num_joints=2)
        self.reset()

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    def joint_state(self) -> tuple[mx.array, mx.array]:
        return self.get_joint_state(None)

    def reset(self, *, soft: bool = False) -> None:
        self.reset_envs(list(range(self.num_envs)))

    def reset_envs(self, env_ids: list[int]) -> None:
        if len(env_ids) == 0:
            return
        joint_pos = mx.zeros((len(env_ids), 2), dtype=mx.float32)
        joint_vel = mx.zeros((len(env_ids), 2), dtype=mx.float32)
        joint_pos[:, 1] = self.reset_sampler.uniform(
            (len(env_ids),),
            self.cfg.initial_pole_angle_range[0],
            self.cfg.initial_pole_angle_range[1],
        )
        self.state.reset_envs(env_ids, joint_pos=joint_pos, joint_vel=joint_vel, joint_effort_target=0.0)

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        cart_pos = self.state.joint_pos[:, 0]
        pole_angle = self.state.joint_pos[:, 1]
        cart_vel = self.state.joint_vel[:, 0]
        pole_vel = self.state.joint_vel[:, 1]
        force = self.state.joint_effort_target[:, 0] * self.cfg.force_mag
        costheta = mx.cos(pole_angle)
        sintheta = mx.sin(pole_angle)
        total_mass = self.cfg.mass_cart + self.cfg.mass_pole
        polemass_length = self.cfg.mass_pole * self.cfg.pole_half_length

        temp = (force + polemass_length * mx.square(pole_vel) * sintheta) / total_mass
        theta_acc = (
            self.cfg.gravity * sintheta - costheta * temp
        ) / (
            self.cfg.pole_half_length
            * (4.0 / 3.0 - self.cfg.mass_pole * mx.square(costheta) / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        dt = self.physics_dt
        self.state.joint_pos[:, 0] = cart_pos + dt * cart_vel
        self.state.joint_vel[:, 0] = cart_vel + dt * x_acc
        self.state.joint_pos[:, 1] = pole_angle + dt * pole_vel
        self.state.joint_vel[:, 1] = pole_vel + dt * theta_acc

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        return self.state.read()

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Any | None = None,
    ) -> None:
        del articulation
        self.state.set_effort_target(efforts, joint_ids=joint_ids or [0])

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

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Any | None = None) -> None:
        del articulation, root_pose, env_ids

    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Any | None = None,
    ) -> None:
        del articulation, root_velocity, env_ids

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "num_envs": self.num_envs,
            "capabilities": self.capabilities.__dict__,
            "contract": {
                "reset_signature": self.contract.reset_signature,
                "step_signature": self.contract.step_signature,
                "articulations": self.contract.articulations.__dict__,
            },
            "subsystems": {
                "terrain": False,
                "contacts": False,
                "deterministic_resets": True,
                "rollout_helpers": True,
            },
            "joint_state_shape": list(self.state.joint_pos.shape),
        }


class MacCartpoleEnv:
    """A vectorized cartpole environment that mirrors the upstream IsaacLab task semantics."""

    def __init__(self, cfg: MacCartpoleEnvCfg | None = None):
        self.cfg = cfg or MacCartpoleEnvCfg()
        mx.random.seed(self.cfg.seed)
        self.reset_sampler = DeterministicResetSampler(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.physics_dt = self.cfg.sim_dt
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.single_action_space = self.cfg.action_space
        self.single_observation_space = {"policy": self.cfg.observation_space}

        self.sim_backend = MacCartpoleSimBackend(self.cfg, reset_sampler=self.reset_sampler.fork("sim-backend"))
        self.actions = mx.zeros((self.num_envs, 1), dtype=mx.float32)
        self.reward_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.reset_terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_time_outs = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.reset_buf = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self.episode_length_buf = mx.zeros((self.num_envs,), dtype=mx.int32)
        self.episode_return_buf = mx.zeros((self.num_envs,), dtype=mx.float32)
        self.common_step_counter = 0
        self.obs_buf = {"policy": mx.zeros((self.num_envs, 4), dtype=mx.float32)}
        self.reset()

    def reset(self) -> tuple[dict[str, mx.array], dict[str, Any]]:
        env_ids = list(range(self.num_envs))
        self._reset_idx(env_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, {}

    def step(self, action: Any) -> tuple[dict[str, mx.array], mx.array, mx.array, mx.array, dict[str, Any]]:
        self._pre_physics_step(action)

        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.sim_backend.step(render=False)

        self.episode_length_buf = self.episode_length_buf + 1
        self.common_step_counter += 1

        joint_pos, joint_vel = self.sim_backend.joint_state()
        cart_pos = joint_pos[:, 0]
        pole_pos = joint_pos[:, 1]
        cart_vel = joint_vel[:, 0]
        pole_vel = joint_vel[:, 1]

        self.reset_terminated = (mx.abs(cart_pos) > self.cfg.max_cart_pos) | (mx.abs(pole_pos) > math.pi / 2.0)
        self.reset_time_outs = self.episode_length_buf >= (self.max_episode_length - 1)
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        self.reward_buf = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            pole_pos,
            pole_vel,
            cart_pos,
            cart_vel,
            self.reset_terminated,
        )
        self.episode_return_buf = self.episode_return_buf + self.reward_buf

        reset_ids = [i for i, flag in enumerate(self.reset_buf.tolist()) if flag]
        extras: dict[str, Any] = {}
        if reset_ids:
            completed_returns = [self.episode_return_buf[i].item() for i in reset_ids]
            completed_lengths = [int(self.episode_length_buf[i].item()) for i in reset_ids]
            extras = {
                "completed_returns": completed_returns,
                "completed_lengths": completed_lengths,
            }
            self._reset_idx(reset_ids)

        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, extras

    def _pre_physics_step(self, action: Any) -> None:
        action = mx.array(action, dtype=mx.float32).reshape((self.num_envs, 1))
        self.actions = mx.clip(action, -1.0, 1.0)

    def _apply_action(self) -> None:
        target = self.cfg.action_scale * self.actions[:, 0]
        self.sim_backend.set_joint_effort_target(None, target, joint_ids=[0])

    def _reset_idx(self, env_ids: list[int]) -> None:
        self.sim_backend.reset_envs(env_ids)
        self.episode_length_buf[mx.array(env_ids)] = 0
        self.episode_return_buf[mx.array(env_ids)] = 0.0

    def _joint_state(self) -> tuple[mx.array, mx.array]:
        return self.sim_backend.joint_state()

    def _get_observations(self) -> dict[str, mx.array]:
        joint_pos, joint_vel = self._joint_state()
        obs = mx.stack(
            [
                joint_pos[:, 1],
                joint_vel[:, 1],
                joint_pos[:, 0],
                joint_vel[:, 0],
            ],
            axis=-1,
        )
        return {"policy": obs}


class MacCartpolePolicy(nn.Module):
    """Shared policy/value MLP for the MLX cartpole baseline."""

    def __init__(self, obs_dim: int = 4, hidden_dim: int = 128, num_actions: int = 2):
        super().__init__()
        self.backbone = [
            nn.Linear(obs_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ]
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, obs: mx.array) -> tuple[mx.array, mx.array]:
        x = obs
        for layer in self.backbone:
            x = mx.tanh(layer(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: mx.array,
    pole_vel: mx.array,
    cart_pos: mx.array,
    cart_vel: mx.array,
    reset_terminated: mx.array,
) -> mx.array:
    """Mirror the upstream IsaacLab cartpole reward structure with MLX arrays."""

    terminated = reset_terminated.astype(mx.float32)
    rew_alive = rew_scale_alive * (1.0 - terminated)
    rew_termination = rew_scale_terminated * terminated
    rew_pole_pos = rew_scale_pole_pos * mx.square(pole_pos)
    rew_cart_vel = rew_scale_cart_vel * mx.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * mx.abs(pole_vel)
    return rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel


def _categorical_log_probs(logits: mx.array, actions: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(log_probs, actions.reshape((-1, 1)), axis=1).squeeze(-1)


def _categorical_entropy(logits: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.softmax(logits, axis=-1)
    return -mx.sum(probs * log_probs, axis=-1)


def _action_index_to_force(actions: mx.array) -> mx.array:
    return mx.where(actions == 0, -1.0, 1.0).reshape((-1, 1))


def _ppo_loss(
    model: MacCartpolePolicy,
    obs: mx.array,
    actions: mx.array,
    old_log_probs: mx.array,
    advantages: mx.array,
    returns: mx.array,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> mx.array:
    logits, values = model(obs)
    log_probs = _categorical_log_probs(logits, actions)
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -mx.mean(mx.minimum(ratio * advantages, clipped_ratio * advantages))
    value_loss = 0.5 * mx.mean(mx.square(returns - values))
    entropy_bonus = mx.mean(_categorical_entropy(logits))
    return policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus


def train_cartpole_policy(cfg: MacCartpoleTrainCfg) -> dict[str, Any]:
    """Train a discrete-action cartpole policy that drives the continuous upstream-style force input."""

    cfg.hidden_dim = resolve_resume_hidden_dim(cfg.resume_from, cfg.hidden_dim)
    mx.random.seed(cfg.env.seed)
    env = MacCartpoleEnv(cfg.env)
    model = MacCartpolePolicy(obs_dim=cfg.env.observation_space, hidden_dim=cfg.hidden_dim)
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
        dones_rollout = []
        values_rollout = []

        for _ in range(cfg.rollout_steps):
            logits, values = model(obs)
            action_ids = mx.random.categorical(logits).astype(mx.int32)
            continuous_actions = _action_index_to_force(action_ids)
            next_obs, reward, terminated, truncated, extras = env.step(continuous_actions)
            done = terminated | truncated

            obs_rollout.append(obs)
            actions_rollout.append(action_ids)
            log_probs_rollout.append(_categorical_log_probs(logits, action_ids))
            rewards_rollout.append(reward)
            dones_rollout.append(done.astype(mx.float32))
            values_rollout.append(values)
            obs = next_obs["policy"]

            if extras.get("completed_returns"):
                completed_returns.extend(extras["completed_returns"])

        _, bootstrap_value = model(obs)
        rewards_t = mx.stack(rewards_rollout)
        dones_t = mx.stack(dones_rollout)
        values_t = mx.stack(values_rollout)
        next_values_t = mx.concatenate([values_t[1:], bootstrap_value[None, :]], axis=0)
        advantages, returns = compute_gae(
            rewards_t,
            dones_t,
            values_t,
            next_values_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1,))
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
            )
            optimizer.update(model, grads)
            mx.eval(loss, model.state, optimizer.state)

        if (update + 1) % cfg.eval_interval == 0 or update == 0 or update == cfg.updates - 1:
            mean_reward = mx.mean(rewards_t).item()
            mean_return = mean_recent_return(completed_returns)
            print(
                f"[mlx-cartpole] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path, metadata_path = save_policy_checkpoint(
        model,
        cfg.checkpoint_path,
        build_checkpoint_metadata(
            hidden_dim=cfg.hidden_dim,
            observation_space=cfg.env.observation_space,
            action_space=cfg.env.action_space,
            task_id="Isaac-Cartpole-Direct-v0",
            policy_distribution="categorical",
            policy_action_space=2,
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


def play_cartpole_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacCartpoleEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained cartpole policy greedily and return episode returns."""
    cfg = env_cfg or MacCartpoleEnvCfg(num_envs=1)
    return play_categorical_policy_checkpoint(
        checkpoint_path,
        env_factory=MacCartpoleEnv,
        env_cfg=cfg,
        model_factory=lambda obs_dim, policy_hidden_dim: MacCartpolePolicy(
            obs_dim=obs_dim,
            hidden_dim=policy_hidden_dim,
        ),
        action_transform=_action_index_to_force,
        default_hidden_dim=128,
        episodes=episodes,
        hidden_dim=hidden_dim,
    )
