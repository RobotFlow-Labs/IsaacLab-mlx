# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark-suite contract tests for the mac-native MLX task slices."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

from isaaclab.backends import build_benchmark_dashboard, build_benchmark_trend
from isaaclab.backends.supported_tasks import benchmark_task_groups
from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim.hotpath import (  # noqa: E402
    get_franka_hotpath_backend,
    get_franka_stack_hotpath_backend,
    get_franka_stack_rgb_hotpath_backend,
    get_locomotion_hotpath_backend,
    get_ur10e_hotpath_backend,
)
from isaaclab.backends.mac_sim import (  # noqa: E402
    MacFrankaBinStackEnv,
    MacFrankaBinStackEnvCfg,
    MacFrankaCabinetEnv,
    MacFrankaCabinetEnvCfg,
    MacFrankaOpenDrawerEnv,
    MacFrankaOpenDrawerEnvCfg,
    MacFrankaStackEnv,
    MacFrankaStackEnvCfg,
    MacFrankaStackRgbEnv,
    MacFrankaStackRgbEnvCfg,
    MacUR10LongSuctionStackEnv,
    MacUR10LongSuctionStackEnvCfg,
    MacUR10eDeployReachEnv,
    MacUR10eDeployReachEnvCfg,
)


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[4] / "scripts" / "benchmarks" / "mlx" / "benchmark_mac_tasks.py"
    spec = importlib.util.spec_from_file_location("benchmark_mac_tasks", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_mac_native_benchmark_group_is_stable():
    """The benchmark suite should keep all current mac-native slices in one named task group."""
    benchmark_module = _load_benchmark_module()
    expected_groups = benchmark_task_groups()

    assert benchmark_module.CURRENT_MAC_NATIVE_TASKS == (
        "cartpole",
        "cart-double-pendulum",
        "quadcopter",
        "anymal-c-flat",
        "anymal-c-rough",
        "h1-flat",
        "h1-rough",
        "franka-reach",
        "openarm-reach",
        "openarm-bi-reach",
        "ur10-reach",
        "ur10e-deploy-reach",
        "ur10e-gear-assembly-2f140",
        "ur10e-gear-assembly-2f85",
        "ur10-long-suction-stack",
        "ur10-short-suction-stack",
        "franka-lift",
        "openarm-lift",
        "franka-teddy-bear-lift",
        "franka-stack-instance-randomize",
        "franka-stack",
        "franka-stack-rgb",
        "franka-bin-stack",
        "franka-cabinet",
        "franka-open-drawer",
        "openarm-open-drawer",
    )
    assert benchmark_module.TASK_GROUPS == expected_groups
    assert benchmark_module.resolve_requested_tasks(None, "current-mac-native") == benchmark_module.CURRENT_MAC_NATIVE_TASKS
    assert benchmark_module.resolve_requested_tasks(None, "sensor-mac-native") == (
        "cartpole-rgb-camera",
        "cartpole-depth-camera",
        "anymal-c-flat-height-scan",
        "h1-flat-height-scan",
    )
    assert benchmark_module.resolve_requested_tasks(None, "training-mac-native") == ("train-cartpole",)


def test_run_benchmarks_covers_all_current_mac_native_tasks(tmp_path: Path):
    """The benchmark harness should emit diagnostics for every current mac-native task slice."""
    benchmark_module = _load_benchmark_module()
    expected_franka_hotpath = get_franka_hotpath_backend()
    expected_franka_stack_hotpath = get_franka_stack_hotpath_backend()
    expected_franka_stack_rgb_hotpath = get_franka_stack_rgb_hotpath_backend()
    expected_locomotion_hotpath = get_locomotion_hotpath_backend()
    expected_ur10e_hotpath = get_ur10e_hotpath_backend()

    results = benchmark_module.run_benchmarks(
        benchmark_module.CURRENT_MAC_NATIVE_TASKS,
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=7,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )

    assert results["task_group"] == "current-mac-native"
    assert results["tasks"] == list(benchmark_module.CURRENT_MAC_NATIVE_TASKS)
    assert results["cpu_fallback_detected"] is False
    assert results["cpu_fallback_tasks"] == []
    assert [benchmark["task"] for benchmark in results["benchmarks"]] == list(benchmark_module.CURRENT_MAC_NATIVE_TASKS)

    for benchmark in results["benchmarks"]:
        assert benchmark["runtime"]["compute_backend"] == "mlx"
        assert benchmark["runtime"]["sim_backend"] == "mac-sim"
        assert benchmark["cpu_fallback"]["detected"] is False
        assert benchmark["cpu_fallback"]["active_kernel_backend"] == "metal"
        assert "diagnostics" in benchmark
        assert benchmark["diagnostics"]["sim_backend"]["backend"] == "mac-sim"
        if benchmark["task"] == "cartpole":
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
            }
        elif benchmark["task"] == "cart-double-pendulum":
            assert benchmark["output_signature"].keys() == {
                "final_cart_obs_mean",
                "final_cart_obs_std",
                "final_pendulum_obs_mean",
                "final_pendulum_obs_std",
                "final_cart_reward_mean",
                "final_pendulum_reward_mean",
            }
        elif benchmark["task"] == "quadcopter":
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_root_height_mean",
                "final_distance_to_goal_mean",
            }
        elif benchmark["task"] in {"anymal-c-flat", "anymal-c-rough", "h1-flat", "h1-rough"}:
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_locomotion_hotpath
            expected = {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_root_height_mean",
                "final_root_lin_vel_norm_mean",
                "final_root_ang_vel_norm_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_joint_acc_abs_mean",
                "final_applied_torque_abs_mean",
                "final_contact_count",
            }
            if benchmark["task"] in {"anymal-c-rough", "h1-rough"}:
                expected |= {"height_scan_hit_ratio", "height_scan_mean", "height_scan_std"}
            assert benchmark["output_signature"].keys() == expected
        elif benchmark["task"] == "franka-reach":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_target_distance_mean",
            }
        elif benchmark["task"] == "openarm-reach":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_target_distance_mean",
            }
        elif benchmark["task"] == "openarm-bi-reach":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_left_ee_height_mean",
                "final_right_ee_height_mean",
                "final_left_target_distance_mean",
                "final_right_target_distance_mean",
            }
        elif benchmark["task"] == "ur10-reach":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_ur10e_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_target_distance_mean",
                "final_orientation_error_mean",
            }
        elif benchmark["task"] == "ur10e-deploy-reach":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_ur10e_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_target_distance_mean",
                "final_orientation_error_mean",
            }
        elif benchmark["task"] in {"ur10e-gear-assembly-2f140", "ur10e-gear-assembly-2f85"}:
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_ur10e_hotpath
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["gear_assembly_logic"] is True
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["gear_type_randomization"] is True
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_target_distance_mean",
                "final_orientation_error_mean",
                "final_insert_depth_mean",
                "final_assembled_ratio",
                "final_gear_type_mean",
            }
        elif benchmark["task"] in {"ur10-long-suction-stack", "ur10-short-suction-stack"}:
            assert benchmark["diagnostics"]["sim_backend"]["task"] == benchmark["task"]
            assert benchmark["diagnostics"]["sim_backend"]["upstream_task_id"] in {
                "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
                "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
            }
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_ur10e_hotpath
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["suction_grasp_surrogate"] is True
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_support_cube_height_mean",
                "final_middle_stack_distance_mean",
                "final_top_stack_distance_mean",
                "final_middle_stacked_ratio",
                "final_top_stacked_ratio",
                "final_active_is_top_ratio",
            }
        elif benchmark["task"] == "franka-lift":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_cube_distance_mean",
                "final_cube_height_mean",
                "final_grasp_ratio",
            }
        elif benchmark["task"] == "openarm-lift":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_cube_distance_mean",
                "final_cube_height_mean",
                "final_grasp_ratio",
            }
        elif benchmark["task"] == "franka-teddy-bear-lift":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["manipulated_object"] == "teddy-bear"
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_cube_distance_mean",
                "final_cube_height_mean",
                "final_grasp_ratio",
                "final_lift_gap_mean",
            }
        elif benchmark["task"] == "franka-stack-instance-randomize":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_stack_hotpath
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["instance_randomization"] is True
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_cube_distance_mean",
                "final_cube_height_mean",
                "final_grasp_ratio",
                "final_support_cube_height_mean",
                "final_stack_distance_mean",
                "final_stacked_ratio",
                "final_support_variant_mean",
                "final_movable_variant_mean",
            }
        elif benchmark["task"] == "franka-stack":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_stack_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_cube_distance_mean",
                "final_cube_height_mean",
                "final_grasp_ratio",
                "final_support_cube_height_mean",
                "final_stack_distance_mean",
                "final_stacked_ratio",
            }
        elif benchmark["task"] == "franka-stack-rgb":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_stack_rgb_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_support_cube_height_mean",
                "final_middle_stack_distance_mean",
                "final_top_stack_distance_mean",
                "final_middle_stacked_ratio",
                "final_top_stacked_ratio",
                "final_active_is_top_ratio",
            }
        elif benchmark["task"] == "franka-bin-stack":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_stack_rgb_hotpath
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["bin_stack_logic"] is True
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_support_cube_height_mean",
                "final_middle_stack_distance_mean",
                "final_top_stack_distance_mean",
                "final_middle_stacked_ratio",
                "final_top_stacked_ratio",
                "final_active_is_top_ratio",
                "final_bin_anchor_height_mean",
                "final_support_bin_offset_mean",
            }
        elif benchmark["task"] == "franka-cabinet":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_handle_distance_mean",
                "final_drawer_open_mean",
                "final_drawer_open_ratio",
                "final_drawer_opened_ratio",
            }
        elif benchmark["task"] == "franka-open-drawer":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_handle_distance_mean",
                "final_drawer_open_mean",
                "final_drawer_open_ratio",
                "final_drawer_opened_ratio",
            }
        elif benchmark["task"] == "openarm-open-drawer":
            assert benchmark["diagnostics"]["sim_backend"]["subsystems"]["hotpath"] == expected_franka_hotpath
            assert benchmark["output_signature"].keys() == {
                "final_policy_mean",
                "final_policy_std",
                "final_reward_mean",
                "final_joint_pos_abs_mean",
                "final_joint_vel_abs_mean",
                "final_ee_height_mean",
                "final_handle_distance_mean",
                "final_drawer_open_mean",
                "final_drawer_open_ratio",
                "final_drawer_opened_ratio",
            }
        assert all(math.isfinite(value) for value in benchmark["output_signature"].values())


def test_franka_stack_benchmark_signature_uses_terminal_stack_snapshots():
    """Stack benchmark metrics should use terminal pre-reset observations when successes auto-reset in-step."""

    benchmark_module = _load_benchmark_module()
    cfg = MacFrankaStackEnvCfg(num_envs=2, seed=59, episode_length_s=0.5)
    env = MacFrankaStackEnv(cfg)
    env.sim_backend.cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.grasped[:] = True
    env.sim_backend.stacked[:] = False
    env.sim_backend.state.joint_pos[:, 7] = 0.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    next_obs, reward, terminated, truncated, extras = env.step(actions)
    trace = SimpleNamespace(
        initial_observations={"policy": mx.zeros((cfg.num_envs, cfg.observation_space), dtype=mx.float32)},
        observations=[next_obs],
        rewards=[reward],
        terminated=[terminated],
        truncated=[truncated],
        extras=[extras],
    )

    signature = benchmark_module._franka_output_signature(env, trace)

    assert float(mx.mean(env.sim_backend.stacked.astype(mx.float32)).item()) == 0.0
    assert math.isclose(signature["final_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(signature["final_stack_distance_mean"], 0.0, rel_tol=0.0, abs_tol=1e-6)


def test_franka_cabinet_benchmark_signature_uses_terminal_drawer_snapshots():
    """Cabinet benchmark metrics should use terminal pre-reset observations when successes auto-reset in-step."""

    benchmark_module = _load_benchmark_module()
    cfg = MacFrankaCabinetEnvCfg(num_envs=2, seed=63, episode_length_s=0.5)
    env = MacFrankaCabinetEnv(cfg)
    env.sim_backend.grasped_handle[:] = True
    env.sim_backend.drawer_opened[:] = False
    env.sim_backend.drawer_open_amount[:] = 0.0
    env.sim_backend.handle_anchor_pos_w[:, 0] = env.sim_backend.ee_pos_w[:, 0] - (cfg.drawer_success_distance + 0.03)
    env.sim_backend.handle_anchor_pos_w[:, 1] = env.sim_backend.ee_pos_w[:, 1]
    env.sim_backend.handle_anchor_pos_w[:, 2] = env.sim_backend.ee_pos_w[:, 2]
    env.sim_backend.handle_pos_w[:] = env.sim_backend.handle_anchor_pos_w
    env.sim_backend.state.joint_pos[:, 7] = 0.04

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = -1.0
    next_obs, reward, terminated, truncated, extras = env.step(actions)
    trace = SimpleNamespace(
        initial_observations={"policy": mx.zeros((cfg.num_envs, cfg.observation_space), dtype=mx.float32)},
        observations=[next_obs],
        rewards=[reward],
        terminated=[terminated],
        truncated=[truncated],
        extras=[extras],
    )

    signature = benchmark_module._franka_output_signature(env, trace)

    assert float(mx.mean(env.sim_backend.drawer_open_amount).item()) == 0.0
    assert signature["final_drawer_open_mean"] >= cfg.drawer_success_distance
    assert math.isclose(signature["final_drawer_opened_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_franka_stack_rgb_benchmark_signature_uses_terminal_stack_snapshots():
    """Three-cube stack benchmark metrics should use terminal pre-reset observations when successes auto-reset in-step."""

    benchmark_module = _load_benchmark_module()
    cfg = MacFrankaStackRgbEnvCfg(num_envs=2, seed=67, episode_length_s=0.5)
    env = MacFrankaStackRgbEnv(cfg)
    stack_offset = mx.array([0.0, 0.0, cfg.stack_offset_z], dtype=mx.float32)
    grasp_offset = mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)

    env.sim_backend.middle_stacked[:] = True
    env.sim_backend.top_stacked[:] = False
    env.sim_backend.middle_grasped[:] = False
    env.sim_backend.top_grasped[:] = True
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + 2.0 * cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.middle_cube_pos_w[:, :] = env.sim_backend.support_cube_pos_w + stack_offset
    env.sim_backend.top_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + grasp_offset
    env.sim_backend.state.joint_pos[:, 7] = 0.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    next_obs, reward, terminated, truncated, extras = env.step(actions)
    trace = SimpleNamespace(
        initial_observations={"policy": mx.zeros((cfg.num_envs, cfg.observation_space), dtype=mx.float32)},
        observations=[next_obs],
        rewards=[reward],
        terminated=[terminated],
        truncated=[truncated],
        extras=[extras],
    )

    signature = benchmark_module._franka_output_signature(env, trace)

    assert float(mx.mean(env.sim_backend.top_stacked.astype(mx.float32)).item()) == 0.0
    assert math.isclose(signature["final_middle_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(signature["final_top_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert signature["final_top_stack_distance_mean"] <= cfg.stack_z_threshold
    assert math.isclose(signature["final_active_is_top_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_franka_bin_stack_benchmark_signature_uses_terminal_stack_snapshots():
    """Bin-stack benchmark metrics should use terminal pre-reset observations when successes auto-reset in-step."""

    benchmark_module = _load_benchmark_module()
    cfg = MacFrankaBinStackEnvCfg(num_envs=2, seed=71, episode_length_s=0.5)
    env = MacFrankaBinStackEnv(cfg)
    stack_offset = mx.array([0.0, 0.0, cfg.stack_offset_z], dtype=mx.float32)
    grasp_offset = mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)

    env.sim_backend.middle_stacked[:] = True
    env.sim_backend.top_stacked[:] = False
    env.sim_backend.middle_grasped[:] = False
    env.sim_backend.top_grasped[:] = True
    env.sim_backend.bin_anchor_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + 2.0 * cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.bin_anchor_pos_w
    env.sim_backend.middle_cube_pos_w[:, :] = env.sim_backend.support_cube_pos_w + stack_offset
    env.sim_backend.top_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + grasp_offset
    env.sim_backend.state.joint_pos[:, 7] = 0.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    next_obs, reward, terminated, truncated, extras = env.step(actions)
    trace = SimpleNamespace(
        initial_observations={"policy": mx.zeros((cfg.num_envs, cfg.observation_space), dtype=mx.float32)},
        observations=[next_obs],
        rewards=[reward],
        terminated=[terminated],
        truncated=[truncated],
        extras=[extras],
    )

    signature = benchmark_module._franka_output_signature(env, trace)

    assert float(mx.mean(env.sim_backend.top_stacked.astype(mx.float32)).item()) == 0.0
    assert math.isclose(signature["final_middle_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(signature["final_top_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert signature["final_top_stack_distance_mean"] <= cfg.stack_z_threshold
    assert math.isclose(signature["final_active_is_top_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(signature["final_support_bin_offset_mean"], 0.0, rel_tol=0.0, abs_tol=1e-6)


def test_ur10_suction_stack_benchmark_signature_uses_terminal_stack_snapshots():
    """UR10 suction-stack benchmark metrics should use the reduced 40D terminal policy layout."""

    benchmark_module = _load_benchmark_module()
    cfg = MacUR10LongSuctionStackEnvCfg(num_envs=2, seed=73, episode_length_s=0.5)
    env = MacUR10LongSuctionStackEnv(cfg)
    stack_offset = mx.array([0.0, 0.0, cfg.stack_offset_z], dtype=mx.float32)
    grasp_offset = mx.array([0.0, 0.0, -cfg.grasp_offset_z], dtype=mx.float32)

    env.sim_backend.middle_stacked[:] = True
    env.sim_backend.top_stacked[:] = False
    env.sim_backend.middle_grasped[:] = False
    env.sim_backend.top_grasped[:] = True
    env.sim_backend.support_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + mx.array(
        [0.0, 0.0, -(cfg.grasp_offset_z + 2.0 * cfg.stack_offset_z)],
        dtype=mx.float32,
    )
    env.sim_backend.middle_cube_pos_w[:, :] = env.sim_backend.support_cube_pos_w + stack_offset
    env.sim_backend.top_cube_pos_w[:, :] = env.sim_backend.ee_pos_w + grasp_offset
    env.sim_backend.active_cube = "top"
    env.sim_backend.state.joint_pos[:, 6] = 1.0

    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
    actions[:, -1] = 1.0
    next_obs, reward, terminated, truncated, extras = env.step(actions)
    trace = SimpleNamespace(
        initial_observations={"policy": mx.zeros((cfg.num_envs, cfg.observation_space), dtype=mx.float32)},
        observations=[next_obs],
        rewards=[reward],
        terminated=[terminated],
        truncated=[truncated],
        extras=[extras],
    )

    signature = benchmark_module._franka_output_signature(env, trace)

    assert float(mx.mean(env.sim_backend.top_stacked.astype(mx.float32)).item()) == 0.0
    assert math.isclose(signature["final_middle_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(signature["final_top_stacked_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert signature["final_top_stack_distance_mean"] <= cfg.stack_z_threshold
    assert math.isclose(signature["final_active_is_top_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_run_benchmarks_covers_sensor_mac_native_tasks(tmp_path: Path):
    """The sensor benchmark group should exercise the height-scan locomotion variants."""
    benchmark_module = _load_benchmark_module()

    results = benchmark_module.run_benchmarks(
        benchmark_module.resolve_requested_tasks(None, "sensor-mac-native"),
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=11,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )

    assert results["task_group"] == "sensor-mac-native"
    assert results["cpu_fallback_detected"] is False
    assert [benchmark["task"] for benchmark in results["benchmarks"]] == [
        "cartpole-rgb-camera",
        "cartpole-depth-camera",
        "anymal-c-flat-height-scan",
        "h1-flat-height-scan",
    ]

    for benchmark in results["benchmarks"]:
        if benchmark["task"] in {"cartpole-rgb-camera", "cartpole-depth-camera"}:
            assert benchmark["diagnostics"]["sensor"]["implementation"] == "analytic-cartpole-camera"
            assert benchmark["runtime"]["capabilities"]["sensor"]["cameras"] is False
            assert benchmark["runtime"]["capabilities"]["sensor"]["analytic_camera_tasks"] is True
            assert benchmark["image_shape"][:2] == [100, 100]
            assert benchmark["output_signature"]["final_frame_energy"] > 0.0
        else:
            assert benchmark["sensor_scan_dim"] == 9
            assert benchmark["diagnostics"]["sensor"]["implementation"] == "analytic-terrain-raycast"
            assert benchmark["runtime"]["capabilities"]["sensor"]["raycast"] is True
            assert benchmark["output_signature"]["height_scan_hit_ratio"] == 1.0
            assert benchmark["output_signature"]["height_scan_mean"] > 0.0


def test_run_benchmarks_covers_training_mac_native_tasks(tmp_path: Path):
    """The training benchmark group should remain executable as a named projection of the manifest."""
    benchmark_module = _load_benchmark_module()

    results = benchmark_module.run_benchmarks(
        benchmark_module.resolve_requested_tasks(None, "training-mac-native"),
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=17,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )

    assert results["task_group"] == "training-mac-native"
    assert results["cpu_fallback_detected"] is False
    assert [benchmark["task"] for benchmark in results["benchmarks"]] == ["train-cartpole"]
    benchmark = results["benchmarks"][0]
    assert benchmark["runtime"]["compute_backend"] == "mlx"
    assert benchmark["runtime"]["sim_backend"] == "mac-sim"
    assert benchmark["cpu_fallback"]["detected"] is False
    assert benchmark["train_frames_per_s"] > 0.0
    assert benchmark["completed_episodes"] >= 0


def test_benchmark_cli_writes_json_output(tmp_path: Path, monkeypatch):
    """The benchmark CLI should persist its JSON, dashboard, and trend payloads for CI artifact upload."""
    benchmark_module = _load_benchmark_module()
    output_path = tmp_path / "benchmark-results.json"
    dashboard_path = tmp_path / "benchmark-results-dashboard.json"
    trend_path = tmp_path / "benchmark-results-trend.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_mac_tasks.py",
            "--task-group",
            "current-mac-native",
            "--num-envs",
            "8",
            "--steps",
            "8",
            "--json-out",
            str(output_path),
            "--artifact-dir",
            str(tmp_path),
        ],
    )

    assert benchmark_module.main() == 0
    assert output_path.exists()
    assert dashboard_path.exists()
    assert trend_path.exists()
    payload = output_path.read_text(encoding="utf-8")
    assert '"task_group": "current-mac-native"' in payload
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    trend = json.loads(trend_path.read_text(encoding="utf-8"))
    assert dashboard["summary"]["rollout_task_count"] == len(benchmark_module.resolve_requested_tasks(None, "current-mac-native"))
    assert dashboard["summary"]["training_task_count"] == 0
    assert trend["summary"]["task_count"] == len(benchmark_module.resolve_requested_tasks(None, "current-mac-native"))


def test_dashboard_and_trend_cover_multi_task_training_runs(tmp_path: Path):
    """The reporting helpers should summarize mixed rollout and training benchmark suites."""
    benchmark_module = _load_benchmark_module()

    results = benchmark_module.run_benchmarks(
        benchmark_module.resolve_requested_tasks(None, "full"),
        num_envs=8,
        steps=8,
        train_updates=1,
        rollout_steps=4,
        epochs_per_update=1,
        seed=13,
        quadcopter_thrust_action=0.2,
        artifact_dir=tmp_path,
    )
    dashboard = build_benchmark_dashboard(results, hardware_label="m5-max")
    trend = build_benchmark_trend(results, hardware_label="m5-max")
    expected_full = benchmark_module.resolve_requested_tasks(None, "full")

    assert dashboard["hardware_label"] == "m5-max"
    assert dashboard["summary"]["rollout_task_count"] == len([task for task in expected_full if task != "train-cartpole"])
    assert dashboard["summary"]["training_task_count"] == 1
    assert dashboard["summary"]["fastest_rollout"] is not None
    assert dashboard["summary"]["fastest_training"] is not None
    assert {entry["task"] for entry in dashboard["tasks"]} == set(expected_full)

    trend_entries = {entry["task"]: entry for entry in trend["tasks"]}
    assert trend["summary"]["task_count"] == len(expected_full)
    assert trend_entries["train-cartpole"]["kind"] == "training"
    assert trend_entries["train-cartpole"]["completed_episodes"] >= 0
    assert trend_entries["cartpole"]["kind"] == "rollout"
    assert trend_entries["cartpole"]["env_steps_per_s"] > 0.0
