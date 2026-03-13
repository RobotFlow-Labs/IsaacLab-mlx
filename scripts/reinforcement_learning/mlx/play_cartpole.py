# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play the mac-native MLX cartpole baseline from a saved checkpoint."""

from __future__ import annotations

import argparse

from isaaclab.backends.mac_sim import MacCartpoleEnvCfg, play_cartpole_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the MLX/mac-sim cartpole baseline.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    returns = play_cartpole_policy(
        args.checkpoint,
        env_cfg=MacCartpoleEnvCfg(num_envs=1, seed=args.seed),
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
    )
    for index, value in enumerate(returns, start=1):
        print(f"[mlx-cartpole] episode={index} return={value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
