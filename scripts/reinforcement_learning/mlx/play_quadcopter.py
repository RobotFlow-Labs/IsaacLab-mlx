# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the mac-native quadcopter slice with random or constant actions."""

from __future__ import annotations

import argparse

import mlx.core as mx

from isaaclab.backends.mac_sim import MacQuadcopterEnv, MacQuadcopterEnvCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the MLX/mac-sim quadcopter slice.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--random-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thrust-action", type=float, default=0.2)
    parser.add_argument("--roll-action", type=float, default=0.0)
    parser.add_argument("--pitch-action", type=float, default=0.0)
    parser.add_argument("--yaw-action", type=float, default=0.0)
    return parser.parse_args()


def _actions(args: argparse.Namespace, num_envs: int) -> mx.array:
    if args.random_actions:
        return mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 4))
    return mx.array(
        [[args.thrust_action, args.roll_action, args.pitch_action, args.yaw_action]] * num_envs,
        dtype=mx.float32,
    )


def main() -> int:
    args = parse_args()
    cfg = MacQuadcopterEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s)
    env = MacQuadcopterEnv(cfg)
    mx.random.seed(args.seed)
    env.reset()

    completed: list[tuple[int, float]] = []
    for _ in range(args.max_steps):
        _, _, _, _, extras = env.step(_actions(args, cfg.num_envs))
        if not extras.get("completed_lengths"):
            continue

        lengths = extras["completed_lengths"]
        distance = float(extras["final_distance_to_goal"])
        completed.extend((int(length), distance) for length in lengths)
        if len(completed) >= args.episodes:
            break

    for index, (length, distance) in enumerate(completed[: args.episodes], start=1):
        print(f"[mlx-quadcopter] episode={index} length={length} final_distance={distance:.4f}")

    if len(completed) < args.episodes:
        print(
            "[mlx-quadcopter] completed fewer episodes than requested "
            f"({len(completed)}/{args.episodes}) within max_steps={args.max_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
