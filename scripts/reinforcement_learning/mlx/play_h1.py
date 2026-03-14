# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the mac-native H1 flat slice with random, zero, or checkpointed actions."""

from __future__ import annotations

import argparse

import mlx.core as mx

from isaaclab.backends.mac_sim import MacH1FlatEnv, MacH1FlatEnvCfg, play_h1_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the MLX/mac-sim H1 flat slice.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random-actions", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = MacH1FlatEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s)

    if args.checkpoint:
        returns = play_h1_policy(args.checkpoint, env_cfg=cfg, episodes=args.episodes)
        for index, episode_return in enumerate(returns, start=1):
            print(f"[mlx-h1-flat] episode={index} return={episode_return:.4f}")
        if len(returns) < args.episodes:
            print(f"[mlx-h1-flat] completed fewer episodes than requested ({len(returns)}/{args.episodes})")
        return 0

    env = MacH1FlatEnv(cfg)
    env.reset()

    completed: list[tuple[int, float]] = []
    for _ in range(args.max_steps):
        if args.random_actions:
            actions = mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
        else:
            actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
        _, _, _, _, extras = env.step(actions)
        if not extras.get("completed_returns"):
            continue

        lengths = extras["completed_lengths"]
        returns = extras["completed_returns"]
        completed.extend((int(length), float(episode_return)) for length, episode_return in zip(lengths, returns, strict=True))
        if len(completed) >= args.episodes:
            break

    for index, (length, episode_return) in enumerate(completed[: args.episodes], start=1):
        print(f"[mlx-h1-flat] episode={index} length={length} return={episode_return:.4f}")

    if len(completed) < args.episodes:
        print(
            "[mlx-h1-flat] completed fewer episodes than requested "
            f"({len(completed)}/{args.episodes}) within max_steps={args.max_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
