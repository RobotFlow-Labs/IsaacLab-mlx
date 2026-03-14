# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the MLX/mac-sim task slices on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from isaaclab.backends import get_runtime_state
from isaaclab.backends.mac_sim import (
    MacAnymalCFlatEnv,
    MacAnymalCFlatEnvCfg,
    MacCartDoublePendulumEnv,
    MacCartDoublePendulumEnvCfg,
    MacCartpoleEnv,
    MacCartpoleEnvCfg,
    MacCartpoleTrainCfg,
    MacQuadcopterEnv,
    MacQuadcopterEnvCfg,
    mac_env_diagnostics,
    rollout_env,
    train_cartpole_policy,
)

TASK_CHOICES = ("cartpole", "cart-double-pendulum", "quadcopter", "anymal-c-flat", "train-cartpole")


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the MLX/mac-sim task slices.")
    parser.add_argument("--tasks", nargs="+", choices=TASK_CHOICES, default=list(TASK_CHOICES))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--train-updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quadcopter-thrust-action", type=float, default=0.2)
    parser.add_argument("--artifact-dir", type=Path, default=Path("logs/benchmarks/mlx"))
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _sync(values: list[mx.array]) -> None:
    """Force MLX execution for stable timing."""
    mx.eval(*values)


def _make_benchmark_result(
    name: str,
    *,
    num_envs: int,
    steps: int,
    elapsed_s: float,
    runtime_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized benchmark payload."""
    env_steps = num_envs * steps
    result = {
        "task": name,
        "num_envs": num_envs,
        "steps": steps,
        "elapsed_s": elapsed_s,
        "env_steps_per_s": env_steps / elapsed_s,
        "mean_step_ms": (elapsed_s / steps) * 1000.0,
        "runtime": runtime_state,
    }
    if extra:
        result.update(extra)
    return result


def benchmark_cartpole(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the cartpole MLX env step loop."""
    env = MacCartpoleEnv(MacCartpoleEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, 1), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "cartpole",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=get_runtime_state(env.runtime),
        extra={
            "observation_dim": env.cfg.observation_space,
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_cart_double_pendulum(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the MARL cart-double-pendulum step loop."""
    env = MacCartDoublePendulumEnv(MacCartDoublePendulumEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = {
        "cart": mx.zeros((num_envs, 1), dtype=mx.float32),
        "pendulum": mx.zeros((num_envs, 1), dtype=mx.float32),
    }
    _sync([observations["cart"], observations["pendulum"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "cart-double-pendulum",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=get_runtime_state(env.runtime),
        extra={
            "cart_observation_dim": env.cfg.observation_spaces["cart"],
            "pendulum_observation_dim": env.cfg.observation_spaces["pendulum"],
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_quadcopter(num_envs: int, steps: int, seed: int, thrust_action: float) -> dict[str, Any]:
    """Benchmark the quadcopter MLX env step loop."""
    env = MacQuadcopterEnv(MacQuadcopterEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, 4), dtype=mx.float32)
    actions[:, 0] = thrust_action
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "quadcopter",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=get_runtime_state(env.runtime),
        extra={
            "observation_dim": env.cfg.observation_space,
            "thrust_action": thrust_action,
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_anymal_c_flat(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the ANYmal-C flat locomotion MLX env step loop."""
    env = MacAnymalCFlatEnv(MacAnymalCFlatEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "anymal-c-flat",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=get_runtime_state(env.runtime),
        extra={
            "observation_dim": env.cfg.observation_space,
            "action_dim": env.cfg.action_space,
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_train_cartpole(
    num_envs: int,
    updates: int,
    rollout_steps: int,
    epochs_per_update: int,
    seed: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Benchmark the cartpole MLX training loop."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "cartpole_benchmark_policy.npz"
    cfg = MacCartpoleTrainCfg(
        env=MacCartpoleEnvCfg(num_envs=num_envs, seed=seed),
        updates=updates,
        rollout_steps=rollout_steps,
        epochs_per_update=epochs_per_update,
        checkpoint_path=str(checkpoint_path),
        eval_interval=max(1, updates),
    )

    start = time.perf_counter()
    result = train_cartpole_policy(cfg)
    elapsed_s = time.perf_counter() - start
    frames = num_envs * updates * rollout_steps

    benchmark = {
        "task": "train-cartpole",
        "num_envs": num_envs,
        "updates": updates,
        "rollout_steps": rollout_steps,
        "epochs_per_update": epochs_per_update,
        "elapsed_s": elapsed_s,
        "train_frames": frames,
        "train_frames_per_s": frames / elapsed_s,
        "checkpoint_path": result["checkpoint_path"],
        "completed_episodes": result["completed_episodes"],
        "mean_recent_return": result["mean_recent_return"],
    }
    benchmark["runtime"] = get_runtime_state()
    return benchmark


def main() -> int:
    """Run the requested benchmarks and optionally write JSON output."""
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.json_out is None:
        args.json_out = args.artifact_dir / "benchmark-results.json"
    results = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "benchmarks": [],
    }

    for task in args.tasks:
        if task == "cartpole":
            benchmark = benchmark_cartpole(args.num_envs, args.steps, args.seed)
        elif task == "cart-double-pendulum":
            benchmark = benchmark_cart_double_pendulum(args.num_envs, args.steps, args.seed)
        elif task == "quadcopter":
            benchmark = benchmark_quadcopter(args.num_envs, args.steps, args.seed, args.quadcopter_thrust_action)
        elif task == "anymal-c-flat":
            benchmark = benchmark_anymal_c_flat(args.num_envs, args.steps, args.seed)
        else:
            benchmark = benchmark_train_cartpole(
                args.num_envs,
                args.train_updates,
                args.rollout_steps,
                args.epochs_per_update,
                args.seed,
                args.artifact_dir,
            )
        results["benchmarks"].append(benchmark)

    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
