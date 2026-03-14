# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the mac-native cart-double-pendulum MARL slice with random or constant actions."""

from __future__ import annotations

import argparse

import mlx.core as mx

from isaaclab.backends.mac_sim import MacCartDoublePendulumEnv, MacCartDoublePendulumEnvCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the MLX/mac-sim cart-double-pendulum slice.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--random-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cart-action", type=float, default=0.0)
    parser.add_argument("--pendulum-action", type=float, default=0.0)
    return parser.parse_args()


def _actions(args: argparse.Namespace, num_envs: int) -> dict[str, mx.array]:
    if args.random_actions:
        return {
            "cart": mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1)),
            "pendulum": mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1)),
        }
    return {
        "cart": mx.full((num_envs, 1), args.cart_action, dtype=mx.float32),
        "pendulum": mx.full((num_envs, 1), args.pendulum_action, dtype=mx.float32),
    }


def main() -> int:
    args = parse_args()
    cfg = MacCartDoublePendulumEnvCfg(num_envs=args.num_envs, seed=args.seed)
    env = MacCartDoublePendulumEnv(cfg)
    mx.random.seed(args.seed)
    env.reset()

    completed: list[tuple[float, float, int]] = []
    for _ in range(args.max_steps):
        _, _, _, _, extras = env.step(_actions(args, cfg.num_envs))
        if not extras.get("completed_returns"):
            continue

        cart_returns = extras["completed_returns"]["cart"]
        pendulum_returns = extras["completed_returns"]["pendulum"]
        lengths = extras["completed_lengths"]
        completed.extend(zip(cart_returns, pendulum_returns, lengths, strict=True))
        if len(completed) >= args.episodes:
            break

    for index, (cart_return, pendulum_return, length) in enumerate(completed[: args.episodes], start=1):
        print(
            f"[mlx-cart-double-pendulum] episode={index} cart_return={cart_return:.4f} "
            f"pendulum_return={pendulum_return:.4f} length={length}"
        )

    if len(completed) < args.episodes:
        print(
            "[mlx-cart-double-pendulum] completed fewer episodes than requested "
            f"({len(completed)}/{args.episodes}) within max_steps={args.max_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
