# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the mac-native MLX cartpole baseline."""

from __future__ import annotations

import argparse

from isaaclab.backends.mac_sim import (
    MacCartpoleEnvCfg,
    MacCartpoleTrainCfg,
    resolve_resume_hidden_dim,
    train_cartpole_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MLX/mac-sim cartpole baseline.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="logs/mlx/cartpole_policy.npz")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else resolve_resume_hidden_dim(args.resume_from, 128)
    cfg = MacCartpoleTrainCfg(
        env=MacCartpoleEnvCfg(num_envs=args.num_envs, seed=args.seed),
        hidden_dim=hidden_dim,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        epochs_per_update=args.epochs_per_update,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        eval_interval=args.eval_interval,
    )
    result = train_cartpole_policy(cfg)
    print(f"[mlx-cartpole] checkpoint={result['checkpoint_path']}")
    if result["resumed_from"] is not None:
        print(f"[mlx-cartpole] resumed_from={result['resumed_from']}")
    print(f"[mlx-cartpole] completed_episodes={result['completed_episodes']}")
    print(f"[mlx-cartpole] mean_recent_return={result['mean_recent_return']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
