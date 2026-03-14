# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI helpers for the MLX/mac-sim task slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from isaaclab.backends.kernel_inventory import CURRENT_MAC_NATIVE_TASKS
from isaaclab.backends.mac_sim import (
    MacAnymalCFlatEnv,
    MacAnymalCFlatEnvCfg,
    MacAnymalCTrainCfg,
    MacCartDoublePendulumEnv,
    MacCartDoublePendulumEnvCfg,
    MacCartpoleEnvCfg,
    MacCartpoleTrainCfg,
    MacH1FlatEnv,
    MacH1FlatEnvCfg,
    MacH1TrainCfg,
    MacQuadcopterEnv,
    MacQuadcopterEnvCfg,
    play_anymal_c_policy,
    play_cartpole_policy,
    play_h1_policy,
    resolve_resume_hidden_dim,
    train_anymal_c_policy,
    train_cartpole_policy,
    train_h1_policy,
)

EVAL_TASKS = CURRENT_MAC_NATIVE_TASKS
TRAIN_TASKS = ("cartpole", "anymal-c-flat", "h1-flat")

TASK_PREFIXES = {
    "cartpole": "mlx-cartpole",
    "cart-double-pendulum": "mlx-cart-double-pendulum",
    "quadcopter": "mlx-quadcopter",
    "anymal-c-flat": "mlx-anymal-c-flat",
    "h1-flat": "mlx-h1-flat",
}


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manual_episode_loop(
    *,
    env: Any,
    action_factory: Callable[[], Any],
    extract_completed: Callable[[dict[str, Any]], list[dict[str, Any]]],
    episodes: int,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    mx.random.seed(seed)
    env.reset()
    completed: list[dict[str, Any]] = []
    for _ in range(max_steps):
        _, _, _, _, extras = env.step(action_factory())
        completed.extend(extract_completed(extras))
        if len(completed) >= episodes:
            break
    return completed[:episodes]


def _build_train_parser(default_task: str | None) -> argparse.ArgumentParser:
    description = "Train the shared MLX/mac-sim task runner."
    if default_task is not None:
        description = f"Train the MLX/mac-sim {default_task} slice."
    parser = argparse.ArgumentParser(description=description)
    if default_task is None:
        parser.add_argument("--task", choices=TRAIN_TASKS, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--epochs-per-update", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--action-std", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--episode-length-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _build_eval_parser(default_task: str | None) -> argparse.ArgumentParser:
    description = "Evaluate or replay the shared MLX/mac-sim task runner."
    if default_task is not None:
        description = f"Evaluate or replay the MLX/mac-sim {default_task} slice."
    parser = argparse.ArgumentParser(description=description)
    if default_task is None:
        parser.add_argument("--task", choices=EVAL_TASKS, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--random-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cart-action", type=float, default=0.0)
    parser.add_argument("--pendulum-action", type=float, default=0.0)
    parser.add_argument("--thrust-action", type=float, default=0.2)
    parser.add_argument("--roll-action", type=float, default=0.0)
    parser.add_argument("--pitch-action", type=float, default=0.0)
    parser.add_argument("--yaw-action", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _train_task(task: str, args: argparse.Namespace) -> dict[str, Any]:
    if task == "cartpole":
        hidden_dim = args.hidden_dim if args.hidden_dim is not None else resolve_resume_hidden_dim(args.resume_from, 128)
        cfg = MacCartpoleTrainCfg(
            env=MacCartpoleEnvCfg(num_envs=args.num_envs, seed=args.seed),
            hidden_dim=hidden_dim,
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            epochs_per_update=args.epochs_per_update,
            learning_rate=args.learning_rate,
            checkpoint_path=args.checkpoint or "logs/mlx/cartpole_policy.npz",
            resume_from=args.resume_from,
            eval_interval=args.eval_interval,
        )
        result = train_cartpole_policy(cfg)
    elif task == "anymal-c-flat":
        hidden_dim = args.hidden_dim if args.hidden_dim is not None else resolve_resume_hidden_dim(args.resume_from, 128)
        cfg = MacAnymalCTrainCfg(
            env=MacAnymalCFlatEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s),
            hidden_dim=hidden_dim,
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            epochs_per_update=args.epochs_per_update,
            learning_rate=args.learning_rate,
            action_std=0.35 if args.action_std is None else args.action_std,
            checkpoint_path=args.checkpoint or "logs/mlx/anymal_c_flat_policy.npz",
            resume_from=args.resume_from,
            eval_interval=args.eval_interval,
        )
        result = train_anymal_c_policy(cfg)
    elif task == "h1-flat":
        hidden_dim = args.hidden_dim if args.hidden_dim is not None else resolve_resume_hidden_dim(args.resume_from, 192)
        cfg = MacH1TrainCfg(
            env=MacH1FlatEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s),
            hidden_dim=hidden_dim,
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            epochs_per_update=args.epochs_per_update,
            learning_rate=args.learning_rate,
            action_std=0.28 if args.action_std is None else args.action_std,
            checkpoint_path=args.checkpoint or "logs/mlx/h1_flat_policy.npz",
            resume_from=args.resume_from,
            eval_interval=args.eval_interval,
        )
        result = train_h1_policy(cfg)
    else:
        raise ValueError(f"Unsupported MLX training task: {task}")

    payload = {
        "task": task,
        "prefix": TASK_PREFIXES[task],
        "train_cfg": result["train_cfg"],
        "checkpoint_path": result["checkpoint_path"],
        "metadata_path": result["metadata_path"],
        "resumed_from": result["resumed_from"],
        "completed_episodes": result["completed_episodes"],
        "mean_recent_return": float(result["mean_recent_return"]),
    }
    return payload


def _extract_locomotion_completed(extras: dict[str, Any]) -> list[dict[str, Any]]:
    if not extras.get("completed_returns"):
        return []
    lengths = extras["completed_lengths"]
    returns = extras["completed_returns"]
    return [
        {"length": int(length), "return": float(episode_return)}
        for length, episode_return in zip(lengths, returns, strict=True)
    ]


def _evaluate_task(task: str, args: argparse.Namespace) -> dict[str, Any]:
    if task == "cartpole":
        if args.checkpoint is None:
            raise ValueError("Cartpole evaluation requires --checkpoint.")
        returns = play_cartpole_policy(
            args.checkpoint,
            env_cfg=MacCartpoleEnvCfg(num_envs=max(1, args.num_envs), seed=args.seed),
            episodes=args.episodes,
            hidden_dim=args.hidden_dim,
        )
        return {
            "task": task,
            "prefix": TASK_PREFIXES[task],
            "mode": "checkpoint",
            "episodes_requested": args.episodes,
            "episodes_completed": len(returns),
            "checkpoint": args.checkpoint,
            "completed": [{"return": float(value)} for value in returns],
        }

    if task == "anymal-c-flat":
        if args.checkpoint:
            returns = play_anymal_c_policy(
                args.checkpoint,
                env_cfg=MacAnymalCFlatEnvCfg(
                    num_envs=max(1, args.num_envs),
                    seed=args.seed,
                    episode_length_s=args.episode_length_s,
                ),
                episodes=args.episodes,
                hidden_dim=args.hidden_dim,
            )
            return {
                "task": task,
                "prefix": TASK_PREFIXES[task],
                "mode": "checkpoint",
                "episodes_requested": args.episodes,
                "episodes_completed": len(returns),
                "checkpoint": args.checkpoint,
                "completed": [{"return": float(value)} for value in returns],
            }

        cfg = MacAnymalCFlatEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s)
        env = MacAnymalCFlatEnv(cfg)
        completed = _manual_episode_loop(
            env=env,
            action_factory=lambda: (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if args.random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            ),
            extract_completed=_extract_locomotion_completed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        return {
            "task": task,
            "prefix": TASK_PREFIXES[task],
            "mode": "manual",
            "episodes_requested": args.episodes,
            "episodes_completed": len(completed),
            "max_steps": args.max_steps,
            "completed": completed,
        }

    if task == "h1-flat":
        if args.checkpoint:
            returns = play_h1_policy(
                args.checkpoint,
                env_cfg=MacH1FlatEnvCfg(
                    num_envs=max(1, args.num_envs),
                    seed=args.seed,
                    episode_length_s=args.episode_length_s,
                ),
                episodes=args.episodes,
                hidden_dim=args.hidden_dim,
            )
            return {
                "task": task,
                "prefix": TASK_PREFIXES[task],
                "mode": "checkpoint",
                "episodes_requested": args.episodes,
                "episodes_completed": len(returns),
                "checkpoint": args.checkpoint,
                "completed": [{"return": float(value)} for value in returns],
            }

        cfg = MacH1FlatEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s)
        env = MacH1FlatEnv(cfg)
        completed = _manual_episode_loop(
            env=env,
            action_factory=lambda: (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if args.random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            ),
            extract_completed=_extract_locomotion_completed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        return {
            "task": task,
            "prefix": TASK_PREFIXES[task],
            "mode": "manual",
            "episodes_requested": args.episodes,
            "episodes_completed": len(completed),
            "max_steps": args.max_steps,
            "completed": completed,
        }

    if task == "cart-double-pendulum":
        if args.checkpoint:
            raise ValueError("Cart-double-pendulum does not support checkpoint evaluation yet.")
        cfg = MacCartDoublePendulumEnvCfg(num_envs=args.num_envs, seed=args.seed)
        env = MacCartDoublePendulumEnv(cfg)
        completed = _manual_episode_loop(
            env=env,
            action_factory=lambda: {
                "cart": (
                    mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, 1))
                    if args.random_actions
                    else mx.full((cfg.num_envs, 1), args.cart_action, dtype=mx.float32)
                ),
                "pendulum": (
                    mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, 1))
                    if args.random_actions
                    else mx.full((cfg.num_envs, 1), args.pendulum_action, dtype=mx.float32)
                ),
            },
            extract_completed=lambda extras: [
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
            ],
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        return {
            "task": task,
            "prefix": TASK_PREFIXES[task],
            "mode": "manual",
            "episodes_requested": args.episodes,
            "episodes_completed": len(completed),
            "max_steps": args.max_steps,
            "completed": completed,
        }

    if task == "quadcopter":
        if args.checkpoint:
            raise ValueError("Quadcopter does not support checkpoint evaluation yet.")
        cfg = MacQuadcopterEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s)
        env = MacQuadcopterEnv(cfg)
        completed = _manual_episode_loop(
            env=env,
            action_factory=lambda: (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, 4))
                if args.random_actions
                else mx.array(
                    [[args.thrust_action, args.roll_action, args.pitch_action, args.yaw_action]] * cfg.num_envs,
                    dtype=mx.float32,
                )
            ),
            extract_completed=lambda extras: [
                {"length": int(length), "final_distance": float(extras["final_distance_to_goal"])}
                for length in extras.get("completed_lengths", [])
            ],
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        return {
            "task": task,
            "prefix": TASK_PREFIXES[task],
            "mode": "manual",
            "episodes_requested": args.episodes,
            "episodes_completed": len(completed),
            "max_steps": args.max_steps,
            "completed": completed,
        }

    raise ValueError(f"Unsupported MLX evaluation task: {task}")


def _print_train_payload(payload: dict[str, Any]) -> None:
    prefix = payload["prefix"]
    print(f"[{prefix}] checkpoint={payload['checkpoint_path']}")
    if payload["resumed_from"] is not None:
        print(f"[{prefix}] resumed_from={payload['resumed_from']}")
    print(f"[{prefix}] completed_episodes={payload['completed_episodes']}")
    print(f"[{prefix}] mean_recent_return={payload['mean_recent_return']:.4f}")


def _print_eval_payload(payload: dict[str, Any]) -> None:
    prefix = payload["prefix"]
    completed = payload["completed"]
    if payload["task"] == "cart-double-pendulum":
        for index, item in enumerate(completed, start=1):
            print(
                f"[{prefix}] episode={index} cart_return={item['cart_return']:.4f} "
                f"pendulum_return={item['pendulum_return']:.4f} length={item['length']}"
            )
    elif payload["task"] == "quadcopter":
        for index, item in enumerate(completed, start=1):
            print(f"[{prefix}] episode={index} length={item['length']} final_distance={item['final_distance']:.4f}")
    elif payload["mode"] == "manual":
        for index, item in enumerate(completed, start=1):
            print(f"[{prefix}] episode={index} length={item['length']} return={item['return']:.4f}")
    else:
        for index, item in enumerate(completed, start=1):
            print(f"[{prefix}] episode={index} return={item['return']:.4f}")

    if payload["episodes_completed"] < payload["episodes_requested"]:
        max_steps = payload.get("max_steps")
        if max_steps is None:
            print(
                f"[{prefix}] completed fewer episodes than requested "
                f"({payload['episodes_completed']}/{payload['episodes_requested']})"
            )
        else:
            print(
                f"[{prefix}] completed fewer episodes than requested "
                f"({payload['episodes_completed']}/{payload['episodes_requested']}) within max_steps={max_steps}"
            )


def run_train_cli(default_task: str | None = None) -> int:
    parser = _build_train_parser(default_task)
    args = parser.parse_args()
    task = default_task or args.task
    payload = _train_task(task, args)
    _write_json(args.json_out, payload)
    _print_train_payload(payload)
    return 0


def run_eval_cli(default_task: str | None = None) -> int:
    parser = _build_eval_parser(default_task)
    args = parser.parse_args()
    task = default_task or args.task
    payload = _evaluate_task(task, args)
    _write_json(args.json_out, payload)
    _print_eval_payload(payload)
    return 0
