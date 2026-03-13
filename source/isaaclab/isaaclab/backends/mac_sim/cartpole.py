# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native cartpole simulator and MLX training helpers."""

from __future__ import annotations

import json
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


@configclass
class MacCartpoleEnvCfg:
    """Cartpole configuration aligned with the upstream IsaacLab direct task where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 5.0

    action_scale: float = 100.0
    action_space: int = 1
    observation_space: int = 4
    state_space: int = 0

    max_cart_pos: float = 3.0
    initial_pole_angle_range: tuple[float, float] = (-0.25 * math.pi, 0.25 * math.pi)

    rew_scale_alive: float = 1.0
    rew_scale_terminated: float = -2.0
    rew_scale_pole_pos: float = -1.0
    rew_scale_cart_vel: float = -0.01
    rew_scale_pole_vel: float = -0.005

    gravity: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_half_length: float = 0.5
    force_mag: float = 1.0

    seed: int = 42


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

    def __init__(self, cfg: MacCartpoleEnvCfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.cart_pos = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self.cart_vel = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self.pole_angle = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self.pole_vel = mx.zeros((cfg.num_envs,), dtype=mx.float32)
        self._joint_effort_target = mx.zeros((cfg.num_envs,), dtype=mx.float32)
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
        ids = mx.array(env_ids)
        self.cart_pos[ids] = 0.0
        self.cart_vel[ids] = 0.0
        self.pole_vel[ids] = 0.0
        self.pole_angle[ids] = mx.random.uniform(
            low=self.cfg.initial_pole_angle_range[0],
            high=self.cfg.initial_pole_angle_range[1],
            shape=(len(env_ids),),
        )
        self._joint_effort_target[ids] = 0.0

    def step(self, *, render: bool = True, update_fabric: bool = False) -> None:
        del render, update_fabric
        force = self._joint_effort_target * self.cfg.force_mag
        costheta = mx.cos(self.pole_angle)
        sintheta = mx.sin(self.pole_angle)
        total_mass = self.cfg.mass_cart + self.cfg.mass_pole
        polemass_length = self.cfg.mass_pole * self.cfg.pole_half_length

        temp = (force + polemass_length * mx.square(self.pole_vel) * sintheta) / total_mass
        theta_acc = (
            self.cfg.gravity * sintheta - costheta * temp
        ) / (
            self.cfg.pole_half_length
            * (4.0 / 3.0 - self.cfg.mass_pole * mx.square(costheta) / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        dt = self.physics_dt
        self.cart_pos = self.cart_pos + dt * self.cart_vel
        self.cart_vel = self.cart_vel + dt * x_acc
        self.pole_angle = self.pole_angle + dt * self.pole_vel
        self.pole_vel = self.pole_vel + dt * theta_acc

    def get_joint_state(self, articulation: Any) -> tuple[mx.array, mx.array]:
        del articulation
        joint_pos = mx.stack([self.cart_pos, self.pole_angle], axis=-1)
        joint_vel = mx.stack([self.cart_vel, self.pole_vel], axis=-1)
        return joint_pos, joint_vel

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Any | None = None,
    ) -> None:
        del articulation, joint_ids
        efforts = mx.array(efforts, dtype=mx.float32).reshape((-1,))
        self._joint_effort_target = efforts

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
        joint_pos = mx.array(joint_pos, dtype=mx.float32)
        joint_vel = mx.array(joint_vel, dtype=mx.float32)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        ids = mx.array(env_ids)
        self.cart_pos[ids] = joint_pos[:, 0]
        self.pole_angle[ids] = joint_pos[:, 1]
        self.cart_vel[ids] = joint_vel[:, 0]
        self.pole_vel[ids] = joint_vel[:, 1]

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
            "cart_pos": self.cart_pos.tolist(),
            "cart_vel": self.cart_vel.tolist(),
            "pole_angle": self.pole_angle.tolist(),
            "pole_vel": self.pole_vel.tolist(),
        }


class MacCartpoleEnv:
    """A vectorized cartpole environment that mirrors the upstream IsaacLab task semantics."""

    def __init__(self, cfg: MacCartpoleEnvCfg | None = None):
        self.cfg = cfg or MacCartpoleEnvCfg()
        mx.random.seed(self.cfg.seed)
        runtime = set_runtime_selection(resolve_runtime_selection("mlx", "mac-sim", "cpu"))
        self.runtime = runtime
        self.device = runtime.device
        self.num_envs = self.cfg.num_envs
        self.physics_dt = self.cfg.sim_dt
        self.step_dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.step_dt)

        self.sim_backend = MacCartpoleSimBackend(self.cfg)
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
        action = mx.array(action, dtype=mx.float32).reshape((self.num_envs, self.cfg.action_space))
        action = mx.clip(action, -1.0, 1.0)
        self.actions = self.cfg.action_scale * action

        for _ in range(self.cfg.decimation):
            self.sim_backend.set_joint_effort_target(None, self.actions[:, 0], joint_ids=[0])
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

    def _reset_idx(self, env_ids: list[int]) -> None:
        self.sim_backend.reset_envs(env_ids)
        self.episode_length_buf[mx.array(env_ids)] = 0
        self.episode_return_buf[mx.array(env_ids)] = 0.0

    def _get_observations(self) -> dict[str, mx.array]:
        joint_pos, joint_vel = self.sim_backend.joint_state()
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


def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix:
        return checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.json")
    return checkpoint_path.with_suffix(".json")


def _write_checkpoint_metadata(checkpoint_path: Path, cfg: MacCartpoleTrainCfg) -> Path:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    metadata = {
        "hidden_dim": cfg.hidden_dim,
        "observation_space": cfg.env.observation_space,
        "action_space": 2,
        "train_cfg": asdict(cfg),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _read_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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

    mx.random.seed(cfg.env.seed)
    env = MacCartpoleEnv(cfg.env)
    model = MacCartpolePolicy(obs_dim=cfg.env.observation_space, hidden_dim=cfg.hidden_dim)
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _ppo_loss)
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
        advantages = []
        gae = mx.zeros((cfg.env.num_envs,), dtype=mx.float32)

        for step in reversed(range(cfg.rollout_steps)):
            next_value = bootstrap_value if step == cfg.rollout_steps - 1 else values_t[step + 1]
            mask = 1.0 - dones_t[step]
            delta = rewards_t[step] + cfg.gamma * next_value * mask - values_t[step]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages.append(gae)

        advantages = mx.stack(list(reversed(advantages)))
        returns = advantages + values_t

        flat_obs = mx.reshape(mx.stack(obs_rollout), (-1, cfg.env.observation_space))
        flat_actions = mx.reshape(mx.stack(actions_rollout), (-1,))
        flat_log_probs = mx.reshape(mx.stack(log_probs_rollout), (-1,))
        flat_advantages = mx.reshape(advantages, (-1,))
        flat_returns = mx.reshape(returns, (-1,))

        adv_mean = mx.mean(flat_advantages)
        adv_std = mx.sqrt(mx.mean(mx.square(flat_advantages - adv_mean)) + 1e-8)
        flat_advantages = (flat_advantages - adv_mean) / adv_std

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
            mean_return = sum(completed_returns[-10:]) / max(1, min(len(completed_returns), 10))
            print(
                f"[mlx-cartpole] update={update + 1}/{cfg.updates} "
                f"mean_step_reward={mean_reward:.4f} mean_recent_return={mean_return:.4f}"
            )

    checkpoint_path = Path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path))
    metadata_path = _write_checkpoint_metadata(checkpoint_path, cfg)
    return {
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "train_cfg": asdict(cfg),
        "completed_episodes": len(completed_returns),
        "mean_recent_return": sum(completed_returns[-10:]) / max(1, min(len(completed_returns), 10)),
    }


def play_cartpole_policy(
    checkpoint_path: str,
    *,
    env_cfg: MacCartpoleEnvCfg | None = None,
    episodes: int = 3,
    hidden_dim: int | None = None,
) -> list[float]:
    """Run a trained cartpole policy greedily and return episode returns."""

    env = MacCartpoleEnv(env_cfg or MacCartpoleEnvCfg(num_envs=1))
    checkpoint = Path(checkpoint_path)
    metadata = _read_checkpoint_metadata(checkpoint)
    policy_hidden_dim = hidden_dim or int(metadata.get("hidden_dim", 128))
    model = MacCartpolePolicy(obs_dim=env.cfg.observation_space, hidden_dim=policy_hidden_dim)
    model.load_weights(str(checkpoint))
    obs = env.reset()[0]["policy"]
    episode_returns: list[float] = []

    while len(episode_returns) < episodes:
        logits, _ = model(obs)
        action_ids = mx.argmax(logits, axis=-1).astype(mx.int32)
        obs, _, _, _, extras = env.step(_action_index_to_force(action_ids))
        obs = obs["policy"]
        if extras.get("completed_returns"):
            episode_returns.extend(extras["completed_returns"])

    return episode_returns[:episodes]
