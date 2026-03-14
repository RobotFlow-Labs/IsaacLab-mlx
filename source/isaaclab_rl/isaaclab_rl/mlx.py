# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLX-native task wrapper surface for the mac-sim runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx

from isaaclab.backends.kernel_inventory import CURRENT_MAC_NATIVE_TASKS
from isaaclab.backends.mac_sim import (
    MacAnymalCFlatEnv,
    MacAnymalCFlatEnvCfg,
    MacAnymalCRoughEnv,
    MacAnymalCRoughEnvCfg,
    MacAnymalCTrainCfg,
    MacCartDoublePendulumEnv,
    MacCartDoublePendulumEnvCfg,
    MacCartpoleEnvCfg,
    MacCartpoleTrainCfg,
    MacFrankaLiftEnv,
    MacFrankaLiftEnvCfg,
    MacFrankaLiftTrainCfg,
    MacFrankaReachEnv,
    MacFrankaReachEnvCfg,
    MacFrankaReachTrainCfg,
    MacH1FlatEnv,
    MacH1FlatEnvCfg,
    MacH1RoughEnv,
    MacH1RoughEnvCfg,
    MacH1TrainCfg,
    MacQuadcopterEnv,
    MacQuadcopterEnvCfg,
    play_anymal_c_policy,
    play_cartpole_policy,
    play_franka_lift_policy,
    play_franka_reach_policy,
    play_h1_policy,
    resolve_resume_hidden_dim,
    train_anymal_c_policy,
    train_cartpole_policy,
    train_franka_lift_policy,
    train_franka_reach_policy,
    train_h1_policy,
)

TRAINABLE_MLX_TASKS = ("cartpole", "anymal-c-flat", "h1-flat", "franka-reach", "franka-lift")


@dataclass(frozen=True)
class MlxTaskSpec:
    """Stable public task metadata for MLX/mac-sim runners."""

    task: str
    trainable: bool
    default_checkpoint: str | None
    default_hidden_dim: int | None
    default_action_std: float | None = None


MLX_TASK_SPECS = {
    "cartpole": MlxTaskSpec("cartpole", True, "logs/mlx/cartpole_policy.npz", 128),
    "cart-double-pendulum": MlxTaskSpec("cart-double-pendulum", False, None, None),
    "quadcopter": MlxTaskSpec("quadcopter", False, None, None),
    "anymal-c-flat": MlxTaskSpec("anymal-c-flat", True, "logs/mlx/anymal_c_flat_policy.npz", 128, 0.35),
    "anymal-c-rough": MlxTaskSpec("anymal-c-rough", False, None, None),
    "franka-reach": MlxTaskSpec("franka-reach", True, "logs/mlx/franka_reach_policy.npz", 128, 0.25),
    "franka-lift": MlxTaskSpec("franka-lift", True, "logs/mlx/franka_lift_policy.npz", 128, 0.25),
    "h1-flat": MlxTaskSpec("h1-flat", True, "logs/mlx/h1_flat_policy.npz", 192, 0.28),
    "h1-rough": MlxTaskSpec("h1-rough", False, None, None),
}


def _serialize_task_spec(spec: MlxTaskSpec) -> dict[str, Any]:
    """Return a JSON-safe task spec payload for CLI and artifact surfaces."""

    return asdict(spec)


def list_mlx_tasks() -> tuple[str, ...]:
    """Return the currently supported MLX/mac-sim task ids."""

    return CURRENT_MAC_NATIVE_TASKS


def list_trainable_mlx_tasks() -> tuple[str, ...]:
    """Return the MLX/mac-sim task ids with a train surface."""

    return TRAINABLE_MLX_TASKS


def get_mlx_task_spec(task: str) -> MlxTaskSpec:
    """Return the stable MLX task metadata for a supported task."""

    try:
        return MLX_TASK_SPECS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported MLX task: {task}") from exc


def train_mlx_task(
    task: str,
    *,
    num_envs: int = 256,
    updates: int = 10,
    rollout_steps: int = 24,
    epochs_per_update: int = 2,
    learning_rate: float = 3e-4,
    hidden_dim: int | None = None,
    action_std: float | None = None,
    checkpoint: str | None = None,
    resume_from: str | None = None,
    eval_interval: int = 5,
    episode_length_s: float = 20.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a supported MLX/mac-sim task and return a normalized payload."""

    spec = get_mlx_task_spec(task)
    if not spec.trainable:
        raise ValueError(f"Task '{task}' does not expose an MLX training surface.")

    if task == "cartpole":
        resolved_hidden_dim = hidden_dim if hidden_dim is not None else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        cfg = MacCartpoleTrainCfg(
            env=MacCartpoleEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/cartpole_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_cartpole_policy(cfg)
    elif task == "anymal-c-flat":
        resolved_hidden_dim = hidden_dim if hidden_dim is not None else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        cfg = MacAnymalCTrainCfg(
            env=MacAnymalCFlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/anymal_c_flat_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_anymal_c_policy(cfg)
    elif task == "h1-flat":
        resolved_hidden_dim = hidden_dim if hidden_dim is not None else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 192)
        cfg = MacH1TrainCfg(
            env=MacH1FlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/h1_flat_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_h1_policy(cfg)
    elif task == "franka-reach":
        resolved_hidden_dim = hidden_dim if hidden_dim is not None else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        cfg = MacFrankaReachTrainCfg(
            env=MacFrankaReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_reach_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_reach_policy(cfg)
    elif task == "franka-lift":
        resolved_hidden_dim = hidden_dim if hidden_dim is not None else resolve_resume_hidden_dim(
            resume_from, spec.default_hidden_dim or 128
        )
        cfg = MacFrankaLiftTrainCfg(
            env=MacFrankaLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_lift_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_lift_policy(cfg)
    else:
        raise ValueError(f"Unsupported MLX training task: {task}")

    return {
        "task": task,
        "task_spec": _serialize_task_spec(spec),
        "train_cfg": result["train_cfg"],
        "checkpoint_path": result["checkpoint_path"],
        "metadata_path": result["metadata_path"],
        "resumed_from": result["resumed_from"],
        "completed_episodes": result["completed_episodes"],
        "mean_recent_return": float(result["mean_recent_return"]),
    }


def evaluate_mlx_task(
    task: str,
    *,
    num_envs: int = 64,
    episodes: int = 3,
    seed: int = 42,
    episode_length_s: float = 20.0,
    max_steps: int = 10000,
    checkpoint: str | None = None,
    hidden_dim: int | None = None,
    random_actions: bool = False,
    cart_action: float = 0.0,
    pendulum_action: float = 0.0,
    thrust_action: float = 0.2,
    roll_action: float = 0.0,
    pitch_action: float = 0.0,
    yaw_action: float = 0.0,
) -> dict[str, Any]:
    """Evaluate or replay a supported MLX/mac-sim task."""

    if task == "cartpole":
        if checkpoint is None:
            raise ValueError("Cartpole evaluation requires a checkpoint.")
        returns = play_cartpole_policy(
            checkpoint,
            env_cfg=MacCartpoleEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
            episodes=episodes,
            hidden_dim=hidden_dim,
        )
        return {
            "task": task,
            "mode": "checkpoint",
            "episodes_requested": episodes,
            "episodes_completed": len(returns),
            "completed": [{"return": float(value)} for value in returns],
            "checkpoint": checkpoint,
        }

    if task == "anymal-c-flat":
        if checkpoint is not None:
            returns = play_anymal_c_policy(
                checkpoint,
                env_cfg=MacAnymalCFlatEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacAnymalCFlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacAnymalCFlatEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True)
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "h1-flat":
        if checkpoint is not None:
            returns = play_h1_policy(
                checkpoint,
                env_cfg=MacH1FlatEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacH1FlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacH1FlatEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True)
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "h1-rough":
        if checkpoint is not None:
            raise ValueError("Task 'h1-rough' does not expose checkpoint replay on the public MLX wrapper.")
        cfg = MacH1RoughEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacH1RoughEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True)
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "anymal-c-rough":
        if checkpoint is not None:
            raise ValueError("Task 'anymal-c-rough' does not expose checkpoint replay on the public MLX wrapper.")
        cfg = MacAnymalCRoughEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacAnymalCRoughEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True)
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-reach":
        if checkpoint is not None:
            returns = play_franka_reach_policy(
                checkpoint,
                env_cfg=MacFrankaReachEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacFrankaReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaReachEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True)
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-lift":
        if checkpoint is not None:
            returns = play_franka_lift_policy(
                checkpoint,
                env_cfg=MacFrankaLiftEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacFrankaLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaLiftEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True)
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "cart-double-pendulum":
        if checkpoint is not None:
            raise ValueError("Cart-double-pendulum does not expose checkpoint replay on the public MLX wrapper.")
        env = MacCartDoublePendulumEnv(
            MacCartDoublePendulumEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = {
                "cart": (
                    mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1))
                    if random_actions
                    else mx.full((num_envs, 1), cart_action, dtype=mx.float32)
                ),
                "pendulum": (
                    mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1))
                    if random_actions
                    else mx.full((num_envs, 1), pendulum_action, dtype=mx.float32)
                ),
            }
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {
                    "cart_return": float(cart_return),
                    "pendulum_return": float(pendulum_return),
                    "length": int(length),
                }
                for cart_return, pendulum_return, length in zip(
                    extras.get("completed_returns", {}).get("cart", []),
                    extras.get("completed_returns", {}).get("pendulum", []),
                    extras.get("completed_lengths", []),
                    strict=True,
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "quadcopter":
        if checkpoint is not None:
            raise ValueError("Quadcopter does not expose checkpoint replay on the public MLX wrapper.")
        env = MacQuadcopterEnv(MacQuadcopterEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s))
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 4))
                if random_actions
                else mx.array([[thrust_action, roll_action, pitch_action, yaw_action]] * num_envs, dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "final_distance": float(extras["final_distance_to_goal"])}
                for length in extras.get("completed_lengths", [])
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    raise ValueError(f"Unsupported MLX evaluation task: {task}")
