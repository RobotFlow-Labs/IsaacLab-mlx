# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the mac-native MLX H1 flat locomotion smoke path."""

from __future__ import annotations

import argparse

from isaaclab.backends.mac_sim import MacH1FlatEnvCfg, MacH1TrainCfg, resolve_resume_hidden_dim, train_h1_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MLX/mac-sim H1 flat smoke path.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--epochs-per-update", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--action-std", type=float, default=0.28)
    parser.add_argument("--checkpoint", type=str, default="logs/mlx/h1_flat_policy.npz")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--episode-length-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else resolve_resume_hidden_dim(args.resume_from, 192)
    cfg = MacH1TrainCfg(
        env=MacH1FlatEnvCfg(num_envs=args.num_envs, seed=args.seed, episode_length_s=args.episode_length_s),
        hidden_dim=hidden_dim,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        epochs_per_update=args.epochs_per_update,
        learning_rate=args.learning_rate,
        action_std=args.action_std,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        eval_interval=args.eval_interval,
    )
    result = train_h1_policy(cfg)
    print(f"[mlx-h1-flat] checkpoint={result['checkpoint_path']}")
    if result["resumed_from"] is not None:
        print(f"[mlx-h1-flat] resumed_from={result['resumed_from']}")
    print(f"[mlx-h1-flat] completed_episodes={result['completed_episodes']}")
    print(f"[mlx-h1-flat] mean_recent_return={result['mean_recent_return']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
