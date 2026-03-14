# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the MLX/mac-sim task slices on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from isaaclab.backends.kernel_inventory import CURRENT_MAC_NATIVE_TASKS
from isaaclab.backends import (
    build_benchmark_dashboard,
    build_benchmark_trend,
    detect_cpu_fallback,
    get_runtime_state,
)
from isaaclab.backends.mac_sim import (
    DEFAULT_HEIGHT_SCAN_OFFSETS,
    MacAnymalCFlatEnv,
    MacAnymalCFlatEnvCfg,
    MacAnymalCRoughEnv,
    MacAnymalCRoughEnvCfg,
    MacCartpoleCameraEnv,
    MacCartpoleDepthCameraEnvCfg,
    MacCartDoublePendulumEnv,
    MacCartDoublePendulumEnvCfg,
    MacCartpoleEnv,
    MacCartpoleEnvCfg,
    MacCartpoleRGBCameraEnvCfg,
    MacCartpoleTrainCfg,
    MacFrankaLiftEnv,
    MacFrankaLiftEnvCfg,
    MacFrankaReachEnv,
    MacFrankaReachEnvCfg,
    MacH1FlatEnv,
    MacH1FlatEnvCfg,
    MacH1RoughEnv,
    MacH1RoughEnvCfg,
    MacQuadcopterEnv,
    MacQuadcopterEnvCfg,
    mac_env_diagnostics,
    rollout_env,
    train_cartpole_policy,
)

TRAINING_BENCHMARK_TASKS = ("train-cartpole",)
SENSOR_BENCHMARK_TASKS = (
    "cartpole-rgb-camera",
    "cartpole-depth-camera",
    "anymal-c-flat-height-scan",
    "h1-flat-height-scan",
)
TASK_CHOICES = CURRENT_MAC_NATIVE_TASKS + SENSOR_BENCHMARK_TASKS + TRAINING_BENCHMARK_TASKS
TASK_GROUPS = {
    "current-mac-native": CURRENT_MAC_NATIVE_TASKS,
    "sensor-mac-native": SENSOR_BENCHMARK_TASKS,
    "full": TASK_CHOICES,
}


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the MLX/mac-sim task slices.")
    parser.add_argument("--tasks", nargs="+", choices=TASK_CHOICES, default=None)
    parser.add_argument("--task-group", choices=tuple(TASK_GROUPS), default="full")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--train-updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quadcopter-thrust-action", type=float, default=0.2)
    parser.add_argument("--artifact-dir", type=Path, default=Path("logs/benchmarks/mlx"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--dashboard-out", type=Path, default=None)
    parser.add_argument("--trend-out", type=Path, default=None)
    parser.add_argument("--hardware-label", type=str, default=None)
    return parser.parse_args()


def _sync(values: list[mx.array]) -> None:
    """Force MLX execution for stable timing."""
    mx.eval(*values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apple_chip() -> str | None:
    """Resolve the local Apple Silicon chip label when available."""
    if platform.system() != "Darwin":
        return None
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
    cpu_fallback = detect_cpu_fallback(runtime_state)
    result = {
        "task": name,
        "num_envs": num_envs,
        "steps": steps,
        "elapsed_s": elapsed_s,
        "env_steps_per_s": env_steps / elapsed_s,
        "mean_step_ms": (elapsed_s / steps) * 1000.0,
        "runtime": runtime_state,
        "cpu_fallback": cpu_fallback,
    }
    if extra:
        result.update(extra)
    return result


def _env_runtime_state(env: Any) -> dict[str, Any]:
    """Capture runtime metadata while preserving env-specific sim capabilities."""
    runtime_state = get_runtime_state(env.runtime)
    sim_backend = getattr(env, "sim_backend", None)
    if sim_backend is not None:
        runtime_state["sim_backend"] = getattr(sim_backend, "name", runtime_state["sim_backend"])
        runtime_state["capabilities"]["sim"] = sim_backend.capabilities.__dict__.copy()
    return runtime_state


def _locomotion_output_signature(env: Any, trace: Any) -> dict[str, float]:
    """Capture stable numeric signatures for locomotion benchmarks."""
    policy = trace.observations[-1]["policy"] if trace.observations else trace.initial_observations["policy"]
    reward = trace.rewards[-1] if trace.rewards else mx.zeros((env.num_envs,), dtype=mx.float32)
    joint_pos, joint_vel = env.sim_backend.get_joint_state(None)
    root_lin_vel_norm = mx.linalg.norm(env.sim_backend.root_lin_vel_b, axis=1)
    root_ang_vel_norm = mx.linalg.norm(env.sim_backend.root_ang_vel_b, axis=1)
    return {
        "final_policy_mean": float(mx.mean(policy).item()),
        "final_policy_std": float(mx.std(policy).item()),
        "final_reward_mean": float(mx.mean(reward).item()),
        "final_root_height_mean": float(mx.mean(env.sim_backend.root_pos_w[:, 2]).item()),
        "final_root_lin_vel_norm_mean": float(mx.mean(root_lin_vel_norm).item()),
        "final_root_ang_vel_norm_mean": float(mx.mean(root_ang_vel_norm).item()),
        "final_joint_pos_abs_mean": float(mx.mean(mx.abs(joint_pos)).item()),
        "final_joint_vel_abs_mean": float(mx.mean(mx.abs(joint_vel)).item()),
        "final_joint_acc_abs_mean": float(mx.mean(mx.abs(env.sim_backend.joint_acc)).item()),
        "final_applied_torque_abs_mean": float(mx.mean(mx.abs(env.sim_backend.applied_torque)).item()),
        "final_contact_count": float(mx.sum(env.sim_backend.contact_model.contact_mask.astype(mx.float32)).item()),
    }


def _sensor_output_signature(env: Any) -> dict[str, float]:
    """Capture stable numeric signatures for height-scan sensor slices."""
    if getattr(env, "height_scan_sensor", None) is None:
        return {}
    scan = env.height_scan_sensor.height_scan(env.sim_backend.root_pos_w)
    return {
        "height_scan_mean": float(mx.mean(scan["distances"]).item()),
        "height_scan_std": float(mx.std(scan["distances"]).item()),
        "height_scan_hit_ratio": float(mx.mean(scan["hit_mask"].astype(mx.float32)).item()),
    }


def _cartpole_output_signature(env: Any, trace: Any) -> dict[str, float]:
    """Capture a compact semantic signature for the cartpole slice."""
    policy = trace.observations[-1]["policy"] if trace.observations else trace.initial_observations["policy"]
    reward = trace.rewards[-1] if trace.rewards else mx.zeros((env.num_envs,), dtype=mx.float32)
    joint_pos, joint_vel = env.sim_backend.get_joint_state(None)
    return {
        "final_policy_mean": float(mx.mean(policy).item()),
        "final_policy_std": float(mx.std(policy).item()),
        "final_reward_mean": float(mx.mean(reward).item()),
        "final_joint_pos_abs_mean": float(mx.mean(mx.abs(joint_pos)).item()),
        "final_joint_vel_abs_mean": float(mx.mean(mx.abs(joint_vel)).item()),
    }


def _cart_double_pendulum_output_signature(trace: Any) -> dict[str, float]:
    """Capture a compact semantic signature for the cart-double-pendulum slice."""
    cart_obs = trace.observations[-1]["cart"] if trace.observations else trace.initial_observations["cart"]
    pendulum_obs = trace.observations[-1]["pendulum"] if trace.observations else trace.initial_observations["pendulum"]
    reward = trace.rewards[-1] if trace.rewards else {
        "cart": mx.zeros((cart_obs.shape[0],), dtype=mx.float32),
        "pendulum": mx.zeros((pendulum_obs.shape[0],), dtype=mx.float32),
    }
    return {
        "final_cart_obs_mean": float(mx.mean(cart_obs).item()),
        "final_cart_obs_std": float(mx.std(cart_obs).item()),
        "final_pendulum_obs_mean": float(mx.mean(pendulum_obs).item()),
        "final_pendulum_obs_std": float(mx.std(pendulum_obs).item()),
        "final_cart_reward_mean": float(mx.mean(reward["cart"]).item()),
        "final_pendulum_reward_mean": float(mx.mean(reward["pendulum"]).item()),
    }


def _quadcopter_output_signature(env: Any, trace: Any) -> dict[str, float]:
    """Capture a compact semantic signature for the quadcopter slice."""
    policy = trace.observations[-1]["policy"] if trace.observations else trace.initial_observations["policy"]
    reward = trace.rewards[-1] if trace.rewards else mx.zeros((env.num_envs,), dtype=mx.float32)
    distance_to_goal = mx.linalg.norm(env._desired_pos_w - env.sim_backend.root_pos_w, axis=1)
    return {
        "final_policy_mean": float(mx.mean(policy).item()),
        "final_policy_std": float(mx.std(policy).item()),
        "final_reward_mean": float(mx.mean(reward).item()),
        "final_root_height_mean": float(mx.mean(env.sim_backend.root_pos_w[:, 2]).item()),
        "final_distance_to_goal_mean": float(mx.mean(distance_to_goal).item()),
    }


def _franka_output_signature(env: Any, trace: Any) -> dict[str, float]:
    """Capture compact semantic signatures for the mac-native Franka tasks."""

    policy = trace.observations[-1]["policy"] if trace.observations else trace.initial_observations["policy"]
    reward = trace.rewards[-1] if trace.rewards else mx.zeros((env.num_envs,), dtype=mx.float32)
    joint_pos, joint_vel = env.sim_backend.get_joint_state(None)
    payload = {
        "final_policy_mean": float(mx.mean(policy).item()),
        "final_policy_std": float(mx.std(policy).item()),
        "final_reward_mean": float(mx.mean(reward).item()),
        "final_joint_pos_abs_mean": float(mx.mean(mx.abs(joint_pos)).item()),
        "final_joint_vel_abs_mean": float(mx.mean(mx.abs(joint_vel)).item()),
        "final_ee_height_mean": float(mx.mean(env.sim_backend.ee_pos_w[:, 2]).item()),
    }
    if hasattr(env.sim_backend, "cube_pos_w"):
        payload["final_cube_distance_mean"] = float(
            mx.mean(mx.linalg.norm(env.sim_backend.cube_pos_w - env.sim_backend.ee_pos_w, axis=1)).item()
        )
        payload["final_cube_height_mean"] = float(mx.mean(env.sim_backend.cube_pos_w[:, 2]).item())
        payload["final_grasp_ratio"] = float(mx.mean(env.sim_backend.grasped.astype(mx.float32)).item())
    elif hasattr(env.sim_backend, "target_pos_w"):
        payload["final_target_distance_mean"] = float(mx.mean(env.sim_backend.goal_distance()).item())
    return payload


def _cartpole_camera_output_signature(env: Any, reward: mx.array, image: mx.array) -> dict[str, float]:
    """Capture a compact semantic signature for the synthetic cartpole camera slices."""

    joint_pos, _ = env.sim_backend.joint_state()
    return {
        "final_policy_mean": float(mx.mean(image).item()),
        "final_policy_std": float(mx.std(image).item()),
        "final_reward_mean": float(mx.mean(reward).item()),
        "final_cart_pos_abs_mean": float(mx.mean(mx.abs(joint_pos[:, 0])).item()),
        "final_pole_angle_abs_mean": float(mx.mean(mx.abs(joint_pos[:, 1])).item()),
        "final_frame_energy": float(mx.mean(mx.square(image)).item()),
    }


def _benchmark_cartpole_camera_env(name: str, env: Any, *, num_envs: int, steps: int) -> dict[str, Any]:
    """Benchmark a synthetic cartpole camera env without retaining the full image rollout trace."""

    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    reward_total = 0.0
    terminated_total = 0
    truncated_total = 0
    last_obs = observations["policy"]
    last_reward = mx.zeros((num_envs,), dtype=mx.float32)

    start = time.perf_counter()
    for _ in range(steps):
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        _sync([next_obs["policy"], reward, terminated, truncated])
        reward_total += float(mx.mean(reward).item())
        terminated_total += int(mx.sum(terminated.astype(mx.int32)).item())
        truncated_total += int(mx.sum(truncated.astype(mx.int32)).item())
        last_obs = next_obs["policy"]
        last_reward = reward
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        name,
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "image_shape": list(last_obs.shape[1:]),
            "camera_mode": env.camera_mode,
            "action_dim": env.cfg.action_space,
            "output_signature": _cartpole_camera_output_signature(env, last_reward, last_obs),
            "diagnostics": mac_env_diagnostics(
                env,
                rollout_summary={
                    "steps": steps,
                    "reward_total": reward_total,
                    "terminated_total": terminated_total,
                    "truncated_total": truncated_total,
                    "final_observation_shapes": {"policy": list(last_obs.shape)},
                },
            ),
        },
    )


def resolve_requested_tasks(tasks: list[str] | None, task_group: str) -> tuple[str, ...]:
    """Resolve CLI benchmark selection to an ordered task tuple."""
    if tasks:
        return tuple(tasks)
    return TASK_GROUPS[task_group]


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
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.cfg.observation_space,
            "output_signature": _cartpole_output_signature(env, trace),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_cartpole_rgb_camera(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the synthetic RGB cartpole camera env step loop."""

    env = MacCartpoleCameraEnv(MacCartpoleRGBCameraEnvCfg(num_envs=num_envs, seed=seed))
    return _benchmark_cartpole_camera_env("cartpole-rgb-camera", env, num_envs=num_envs, steps=steps)


def benchmark_cartpole_depth_camera(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the synthetic depth cartpole camera env step loop."""

    env = MacCartpoleCameraEnv(MacCartpoleDepthCameraEnvCfg(num_envs=num_envs, seed=seed))
    return _benchmark_cartpole_camera_env("cartpole-depth-camera", env, num_envs=num_envs, steps=steps)


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
        runtime_state=_env_runtime_state(env),
        extra={
            "cart_observation_dim": env.cfg.observation_spaces["cart"],
            "pendulum_observation_dim": env.cfg.observation_spaces["pendulum"],
            "output_signature": _cart_double_pendulum_output_signature(trace),
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
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.cfg.observation_space,
            "thrust_action": thrust_action,
            "output_signature": _quadcopter_output_signature(env, trace),
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
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.cfg.observation_space,
            "action_dim": env.cfg.action_space,
            "output_signature": _locomotion_output_signature(env, trace),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_anymal_c_rough(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the rough ANYmal-C locomotion MLX env step loop."""

    env = MacAnymalCRoughEnv(MacAnymalCRoughEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "anymal-c-rough",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.observation_space,
            "action_dim": env.cfg.action_space,
            "sensor_scan_dim": env.height_scan_dim,
            "output_signature": _locomotion_output_signature(env, trace) | _sensor_output_signature(env),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_franka_reach(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the Franka reach MLX env step loop."""

    env = MacFrankaReachEnv(MacFrankaReachEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "franka-reach",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.cfg.observation_space,
            "action_dim": env.cfg.action_space,
            "output_signature": _franka_output_signature(env, trace),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_franka_lift(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the Franka lift MLX env step loop."""

    env = MacFrankaLiftEnv(MacFrankaLiftEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "franka-lift",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.cfg.observation_space,
            "action_dim": env.cfg.action_space,
            "output_signature": _franka_output_signature(env, trace),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_anymal_c_flat_height_scan(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the ANYmal-C flat locomotion slice with the mac-native height scan enabled."""
    env = MacAnymalCFlatEnv(
        MacAnymalCFlatEnvCfg(
            num_envs=num_envs,
            seed=seed,
            height_scan_enabled=True,
            height_scan_offsets=DEFAULT_HEIGHT_SCAN_OFFSETS,
        )
    )
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "anymal-c-flat-height-scan",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.observation_space,
            "action_dim": env.cfg.action_space,
            "sensor_scan_dim": env.height_scan_dim,
            "output_signature": _locomotion_output_signature(env, trace) | _sensor_output_signature(env),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_h1_flat(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the H1 flat locomotion MLX env step loop."""
    env = MacH1FlatEnv(MacH1FlatEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "h1-flat",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.cfg.observation_space,
            "action_dim": env.cfg.action_space,
            "output_signature": _locomotion_output_signature(env, trace),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_h1_flat_height_scan(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the H1 flat locomotion slice with the mac-native height scan enabled."""
    env = MacH1FlatEnv(
        MacH1FlatEnvCfg(
            num_envs=num_envs,
            seed=seed,
            height_scan_enabled=True,
            height_scan_offsets=DEFAULT_HEIGHT_SCAN_OFFSETS,
        )
    )
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "h1-flat-height-scan",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.observation_space,
            "action_dim": env.cfg.action_space,
            "sensor_scan_dim": env.height_scan_dim,
            "output_signature": _locomotion_output_signature(env, trace) | _sensor_output_signature(env),
            "diagnostics": mac_env_diagnostics(env, rollout_summary=trace.summary()),
        },
    )


def benchmark_h1_rough(num_envs: int, steps: int, seed: int) -> dict[str, Any]:
    """Benchmark the H1 rough locomotion MLX env step loop."""
    env = MacH1RoughEnv(MacH1RoughEnvCfg(num_envs=num_envs, seed=seed))
    observations, _ = env.reset()
    actions = mx.zeros((num_envs, env.cfg.action_space), dtype=mx.float32)
    _sync([observations["policy"]])

    start = time.perf_counter()
    trace = rollout_env(env, actions, steps=steps, sync_callback=_sync)
    elapsed_s = time.perf_counter() - start

    return _make_benchmark_result(
        "h1-rough",
        num_envs=num_envs,
        steps=steps,
        elapsed_s=elapsed_s,
        runtime_state=_env_runtime_state(env),
        extra={
            "observation_dim": env.observation_space,
            "action_dim": env.cfg.action_space,
            "sensor_scan_dim": env.height_scan_dim,
            "output_signature": _locomotion_output_signature(env, trace) | _sensor_output_signature(env),
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
        "metadata_path": result["metadata_path"],
        "completed_episodes": result["completed_episodes"],
        "mean_recent_return": result["mean_recent_return"],
    }
    benchmark["runtime"] = get_runtime_state()
    benchmark["cpu_fallback"] = detect_cpu_fallback(benchmark["runtime"])
    return benchmark


def run_benchmarks(
    tasks: tuple[str, ...],
    *,
    num_envs: int,
    steps: int,
    train_updates: int,
    rollout_steps: int,
    epochs_per_update: int,
    seed: int,
    quadcopter_thrust_action: float,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Run the requested benchmark suite and return the JSON payload."""
    results = {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "apple_chip": _apple_chip(),
        },
        "parameters": {
            "num_envs": num_envs,
            "steps": steps,
            "train_updates": train_updates,
            "rollout_steps": rollout_steps,
            "epochs_per_update": epochs_per_update,
            "seed": seed,
            "quadcopter_thrust_action": quadcopter_thrust_action,
        },
        "task_group": next((name for name, group in TASK_GROUPS.items() if group == tasks), "custom"),
        "tasks": list(tasks),
        "benchmarks": [],
    }

    for task in tasks:
        if task == "cartpole":
            benchmark = benchmark_cartpole(num_envs, steps, seed)
        elif task == "cartpole-rgb-camera":
            benchmark = benchmark_cartpole_rgb_camera(num_envs, steps, seed)
        elif task == "cartpole-depth-camera":
            benchmark = benchmark_cartpole_depth_camera(num_envs, steps, seed)
        elif task == "cart-double-pendulum":
            benchmark = benchmark_cart_double_pendulum(num_envs, steps, seed)
        elif task == "quadcopter":
            benchmark = benchmark_quadcopter(num_envs, steps, seed, quadcopter_thrust_action)
        elif task == "anymal-c-flat":
            benchmark = benchmark_anymal_c_flat(num_envs, steps, seed)
        elif task == "anymal-c-rough":
            benchmark = benchmark_anymal_c_rough(num_envs, steps, seed)
        elif task == "h1-flat":
            benchmark = benchmark_h1_flat(num_envs, steps, seed)
        elif task == "h1-rough":
            benchmark = benchmark_h1_rough(num_envs, steps, seed)
        elif task == "franka-reach":
            benchmark = benchmark_franka_reach(num_envs, steps, seed)
        elif task == "franka-lift":
            benchmark = benchmark_franka_lift(num_envs, steps, seed)
        elif task == "anymal-c-flat-height-scan":
            benchmark = benchmark_anymal_c_flat_height_scan(num_envs, steps, seed)
        elif task == "h1-flat-height-scan":
            benchmark = benchmark_h1_flat_height_scan(num_envs, steps, seed)
        else:
            benchmark = benchmark_train_cartpole(
                num_envs,
                train_updates,
                rollout_steps,
                epochs_per_update,
                seed,
                artifact_dir,
            )
        results["benchmarks"].append(benchmark)

    cpu_fallback_tasks = [benchmark["task"] for benchmark in results["benchmarks"] if benchmark["cpu_fallback"]["detected"]]
    results["cpu_fallback_detected"] = bool(cpu_fallback_tasks)
    results["cpu_fallback_tasks"] = cpu_fallback_tasks
    return results


def main() -> int:
    """Run the requested benchmarks and optionally write JSON output."""
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.json_out is None:
        args.json_out = args.artifact_dir / "benchmark-results.json"
    if args.dashboard_out is None:
        args.dashboard_out = args.json_out.with_name(f"{args.json_out.stem}-dashboard.json")
    if args.trend_out is None:
        args.trend_out = args.json_out.with_name(f"{args.json_out.stem}-trend.json")
    tasks = resolve_requested_tasks(args.tasks, args.task_group)
    results = run_benchmarks(
        tasks,
        num_envs=args.num_envs,
        steps=args.steps,
        train_updates=args.train_updates,
        rollout_steps=args.rollout_steps,
        epochs_per_update=args.epochs_per_update,
        seed=args.seed,
        quadcopter_thrust_action=args.quadcopter_thrust_action,
        artifact_dir=args.artifact_dir,
    )
    dashboard = build_benchmark_dashboard(results, hardware_label=args.hardware_label)
    trend = build_benchmark_trend(results, hardware_label=args.hardware_label)

    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    _write_json(args.json_out, results)
    _write_json(args.dashboard_out, dashboard)
    _write_json(args.trend_out, trend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
