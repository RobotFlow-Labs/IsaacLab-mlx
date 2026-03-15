# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI helpers for the MLX/mac-sim task slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from isaaclab_rl.mlx import (
    evaluate_mlx_task,
    list_mlx_tasks,
    list_trainable_mlx_tasks,
    train_mlx_task,
)

TASK_PREFIXES = {
    "cartpole": "mlx-cartpole",
    "cartpole-rgb-camera": "mlx-cartpole-rgb-camera",
    "cartpole-depth-camera": "mlx-cartpole-depth-camera",
    "cart-double-pendulum": "mlx-cart-double-pendulum",
    "quadcopter": "mlx-quadcopter",
    "anymal-c-flat": "mlx-anymal-c-flat",
    "anymal-c-rough": "mlx-anymal-c-rough",
    "h1-flat": "mlx-h1-flat",
    "h1-rough": "mlx-h1-rough",
    "franka-reach": "mlx-franka-reach",
    "franka-lift": "mlx-franka-lift",
    "franka-stack": "mlx-franka-stack",
    "franka-stack-rgb": "mlx-franka-stack-rgb",
    "franka-cabinet": "mlx-franka-cabinet",
    "franka-open-drawer": "mlx-franka-open-drawer",
}


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_train_parser(default_task: str | None) -> argparse.ArgumentParser:
    description = "Train the shared MLX/mac-sim task runner."
    if default_task is not None:
        description = f"Train the MLX/mac-sim {default_task} slice."
    parser = argparse.ArgumentParser(description=description)
    if default_task is None:
        parser.add_argument("--task", choices=list_trainable_mlx_tasks(), required=True)
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
        parser.add_argument("--task", choices=list_mlx_tasks(), required=True)
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


def _print_train_payload(payload: dict[str, Any]) -> None:
    prefix = TASK_PREFIXES[payload["task"]]
    print(f"[{prefix}] checkpoint={payload['checkpoint_path']}")
    if payload["resumed_from"] is not None:
        print(f"[{prefix}] resumed_from={payload['resumed_from']}")
    print(f"[{prefix}] completed_episodes={payload['completed_episodes']}")
    print(f"[{prefix}] mean_recent_return={payload['mean_recent_return']:.4f}")


def _print_eval_payload(payload: dict[str, Any]) -> None:
    prefix = TASK_PREFIXES[payload["task"]]
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
    payload = train_mlx_task(
        task,
        num_envs=args.num_envs,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        epochs_per_update=args.epochs_per_update,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        action_std=args.action_std,
        checkpoint=args.checkpoint,
        resume_from=args.resume_from,
        eval_interval=args.eval_interval,
        episode_length_s=args.episode_length_s,
        seed=args.seed,
    )
    _write_json(args.json_out, payload)
    _print_train_payload(payload)
    return 0


def run_eval_cli(default_task: str | None = None) -> int:
    parser = _build_eval_parser(default_task)
    args = parser.parse_args()
    task = default_task or args.task
    payload = evaluate_mlx_task(
        task,
        num_envs=args.num_envs,
        episodes=args.episodes,
        seed=args.seed,
        episode_length_s=args.episode_length_s,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint,
        hidden_dim=args.hidden_dim,
        random_actions=args.random_actions,
        cart_action=args.cart_action,
        pendulum_action=args.pendulum_action,
        thrust_action=args.thrust_action,
        roll_action=args.roll_action,
        pitch_action=args.pitch_action,
        yaw_action=args.yaw_action,
    )
    _write_json(args.json_out, payload)
    _print_eval_payload(payload)
    return 0
