# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the shared MLX/mac-sim task CLIs."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from isaaclab.backends.kernel_inventory import CURRENT_MAC_NATIVE_TASKS
from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()

from isaaclab_rl.mlx import evaluate_mlx_task, list_mlx_tasks, list_trainable_mlx_tasks, train_mlx_task  # noqa: E402


def _load_task_support_module():
    module_path = Path(__file__).resolve().parents[4] / "scripts" / "reinforcement_learning" / "mlx" / "_task_support.py"
    spec = importlib.util.spec_from_file_location("mlx_task_support", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MLX task support module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_task_cli_registry_aligns_with_current_mac_native_tasks():
    module = _load_task_support_module()

    assert tuple(module.TASK_PREFIXES) == list_mlx_tasks()
    assert list_trainable_mlx_tasks() == (
        "cartpole",
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
        "factory-peg-insert",
        "factory-gear-mesh",
        "ur10-long-suction-stack",
        "ur10-short-suction-stack",
        "franka-lift",
        "openarm-lift",
        "agibot-place-toy2box",
        "agibot-place-upright-mug",
        "franka-teddy-bear-lift",
        "franka-stack-instance-randomize",
        "franka-stack",
        "franka-stack-rgb",
        "franka-bin-stack",
        "franka-cabinet",
        "franka-open-drawer",
        "openarm-open-drawer",
    )


def test_shared_task_cli_trains_first_locomotion_task(tmp_path: Path):
    checkpoint_path = tmp_path / "anymal_c_flat_policy.npz"

    payload = train_mlx_task(
        "anymal-c-flat",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        learning_rate=3e-4,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=13,
    )

    assert payload["task"] == "anymal-c-flat"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_rough_locomotion_slices(tmp_path: Path):
    anymal_checkpoint = tmp_path / "anymal_c_rough_policy.npz"
    h1_checkpoint = tmp_path / "h1_rough_policy.npz"

    anymal_payload = train_mlx_task(
        "anymal-c-rough",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        learning_rate=3e-4,
        hidden_dim=32,
        checkpoint=str(anymal_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=15,
    )
    h1_payload = train_mlx_task(
        "h1-rough",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        learning_rate=3e-4,
        hidden_dim=64,
        checkpoint=str(h1_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=17,
    )

    assert anymal_payload["task"] == "anymal-c-rough"
    assert Path(anymal_payload["checkpoint_path"]).exists()
    assert Path(anymal_payload["metadata_path"]).exists()
    assert h1_payload["task"] == "h1-rough"
    assert Path(h1_payload["checkpoint_path"]).exists()
    assert Path(h1_payload["metadata_path"]).exists()


def test_shared_task_cli_evaluates_h1_manual_slice():
    payload = evaluate_mlx_task(
        "h1-flat",
        num_envs=8,
        episodes=1,
        seed=17,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
    )

    assert payload["task"] == "h1-flat"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_h1_rough_manual_slice():
    payload = evaluate_mlx_task(
        "h1-rough",
        num_envs=8,
        episodes=1,
        seed=27,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
    )

    assert payload["task"] == "h1-rough"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_cartpole_camera_manual_slices():
    rgb_payload = evaluate_mlx_task(
        "cartpole-rgb-camera",
        num_envs=8,
        episodes=1,
        seed=35,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
    )
    depth_payload = evaluate_mlx_task(
        "cartpole-depth-camera",
        num_envs=8,
        episodes=1,
        seed=39,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
    )

    assert rgb_payload["task"] == "cartpole-rgb-camera"
    assert rgb_payload["mode"] == "manual"
    assert rgb_payload["episodes_completed"] == 1
    assert depth_payload["task"] == "cartpole-depth-camera"
    assert depth_payload["mode"] == "manual"
    assert depth_payload["episodes_completed"] == 1


def test_shared_task_cli_evaluates_franka_reach_manual_slice():
    payload = evaluate_mlx_task(
        "franka-reach",
        num_envs=8,
        episodes=1,
        seed=19,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-reach"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_ur10e_deploy_reach_manual_slice():
    payload = evaluate_mlx_task(
        "ur10e-deploy-reach",
        num_envs=8,
        episodes=1,
        seed=29,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "ur10e-deploy-reach"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_new_reduced_manipulation_manual_slices():
    cases = (
        ("openarm-reach", 30),
        ("openarm-bi-reach", 31),
        ("ur10-reach", 32),
        ("ur10e-gear-assembly-2f140", 33),
        ("ur10e-gear-assembly-2f85", 34),
        ("openarm-lift", 35),
        ("openarm-open-drawer", 36),
    )

    for task, seed in cases:
        payload = evaluate_mlx_task(
            task,
            num_envs=8,
            episodes=1,
            seed=seed,
            episode_length_s=0.5,
            max_steps=512,
            random_actions=False,
        )

        assert payload["task"] == task
        assert payload["mode"] == "manual"
        assert payload["episodes_requested"] == 1
        assert payload["episodes_completed"] == 1
        assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_trains_franka_reach_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_reach_policy.npz"

    payload = train_mlx_task(
        "franka-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=23,
    )

    assert payload["task"] == "franka-reach"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_ur10e_deploy_reach_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "ur10e_deploy_reach_policy.npz"

    payload = train_mlx_task(
        "ur10e-deploy-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=27,
    )

    assert payload["task"] == "ur10e-deploy-reach"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_new_reduced_manipulation_slices(tmp_path: Path):
    cases = (
        ("openarm-reach", 40),
        ("openarm-bi-reach", 41),
        ("ur10-reach", 42),
        ("ur10e-gear-assembly-2f140", 43),
        ("ur10e-gear-assembly-2f85", 44),
        ("openarm-lift", 45),
        ("openarm-open-drawer", 46),
    )

    for task, seed in cases:
        checkpoint_path = tmp_path / f"{task}_policy.npz"
        payload = train_mlx_task(
            task,
            num_envs=8,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            hidden_dim=48 if task == "openarm-bi-reach" else 32,
            checkpoint=str(checkpoint_path),
            eval_interval=1,
            episode_length_s=0.5,
            seed=seed,
        )

        assert payload["task"] == task
        assert Path(payload["checkpoint_path"]).exists()
        assert Path(payload["metadata_path"]).exists()
        assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_franka_lift_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_lift_policy.npz"

    payload = train_mlx_task(
        "franka-lift",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=29,
    )

    assert payload["task"] == "franka-lift"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_franka_teddy_bear_lift_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_teddy_bear_lift_policy.npz"

    payload = train_mlx_task(
        "franka-teddy-bear-lift",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=31,
    )

    assert payload["task"] == "franka-teddy-bear-lift"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_franka_stack_instance_randomize_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_stack_instance_randomize_policy.npz"

    payload = train_mlx_task(
        "franka-stack-instance-randomize",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=33,
    )

    assert payload["task"] == "franka-stack-instance-randomize"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_franka_stack_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_stack_policy.npz"

    payload = train_mlx_task(
        "franka-stack",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=31,
    )

    assert payload["task"] == "franka-stack"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_franka_stack_rgb_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_stack_rgb_policy.npz"

    payload = train_mlx_task(
        "franka-stack-rgb",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=35,
    )

    assert payload["task"] == "franka-stack-rgb"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_trains_franka_bin_stack_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_bin_stack_policy.npz"

    payload = train_mlx_task(
        "franka-bin-stack",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=37,
    )

    assert payload["task"] == "franka-bin-stack"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_evaluates_franka_stack_manual_slice():
    payload = evaluate_mlx_task(
        "franka-stack",
        num_envs=8,
        episodes=1,
        seed=41,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-stack"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_franka_bin_stack_manual_slice():
    payload = evaluate_mlx_task(
        "franka-bin-stack",
        num_envs=8,
        episodes=1,
        seed=49,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-bin-stack"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_franka_teddy_bear_lift_manual_slice():
    payload = evaluate_mlx_task(
        "franka-teddy-bear-lift",
        num_envs=8,
        episodes=1,
        seed=43,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-teddy-bear-lift"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_franka_stack_instance_randomize_manual_slice():
    payload = evaluate_mlx_task(
        "franka-stack-instance-randomize",
        num_envs=8,
        episodes=1,
        seed=45,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-stack-instance-randomize"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_evaluates_franka_stack_rgb_manual_slice():
    payload = evaluate_mlx_task(
        "franka-stack-rgb",
        num_envs=8,
        episodes=1,
        seed=41,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-stack-rgb"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_trains_franka_cabinet_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_cabinet_policy.npz"

    payload = train_mlx_task(
        "franka-cabinet",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=33,
    )

    assert payload["task"] == "franka-cabinet"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_evaluates_franka_cabinet_manual_slice():
    payload = evaluate_mlx_task(
        "franka-cabinet",
        num_envs=8,
        episodes=1,
        seed=45,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-cabinet"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_trains_franka_open_drawer_slice(tmp_path: Path):
    checkpoint_path = tmp_path / "franka_open_drawer_policy.npz"

    payload = train_mlx_task(
        "franka-open-drawer",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=35,
    )

    assert payload["task"] == "franka-open-drawer"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_evaluates_franka_open_drawer_manual_slice():
    payload = evaluate_mlx_task(
        "franka-open-drawer",
        num_envs=8,
        episodes=1,
        seed=47,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
    )

    assert payload["task"] == "franka-open-drawer"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_shared_task_cli_writes_json_safe_train_payload(tmp_path: Path):
    module = _load_task_support_module()
    checkpoint_path = tmp_path / "cartpole_wrapper_policy.npz"
    output_path = tmp_path / "train_payload.json"

    payload = train_mlx_task(
        "cartpole",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=21,
    )

    module._write_json(output_path, payload)
    stored = json.loads(output_path.read_text(encoding="utf-8"))

    assert stored["task"] == "cartpole"
    assert stored["task_spec"]["task"] == "cartpole"
    assert stored["task_spec"]["trainable"] is True
    assert stored["train_cfg"]["env"]["episode_length_s"] == 0.5
    assert stored["checkpoint_path"].endswith("cartpole_wrapper_policy.npz")


def test_franka_stack_rgb_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    train_script = repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_franka_stack_rgb.py"
    play_script = repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_franka_stack_rgb.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in (train_script, play_script):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_franka_bin_stack_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_franka_bin_stack.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_franka_bin_stack.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_new_reduced_manipulation_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        "train_openarm_reach.py",
        "play_openarm_reach.py",
        "train_openarm_bi_reach.py",
        "play_openarm_bi_reach.py",
        "train_ur10_reach.py",
        "play_ur10_reach.py",
        "train_openarm_lift.py",
        "play_openarm_lift.py",
        "train_openarm_open_drawer.py",
        "play_openarm_open_drawer.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script_name in scripts:
        script = repo_root / "scripts" / "reinforcement_learning" / "mlx" / script_name
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_franka_open_drawer_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_franka_open_drawer.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_franka_open_drawer.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_ur10e_deploy_reach_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_ur10e_deploy_reach.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_ur10e_deploy_reach.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_franka_teddy_bear_lift_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_franka_teddy_bear_lift.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_franka_teddy_bear_lift.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_franka_stack_instance_randomize_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_franka_stack_instance_randomize.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_franka_stack_instance_randomize.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_rough_locomotion_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_anymal_c_rough.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_anymal_c_rough.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_h1_rough.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_h1_rough.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_factory_gear_mesh_thin_wrappers_are_directly_runnable():
    repo_root = Path(__file__).resolve().parents[4]
    scripts = (
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "train_factory_gear_mesh.py",
        repo_root / "scripts" / "reinforcement_learning" / "mlx" / "play_factory_gear_mesh.py",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_factory_peg_insert_benchmark_group_is_runnable(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[4]
    json_out = tmp_path / "factory-peg-insert-benchmark.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmarks" / "mlx" / "benchmark_mac_tasks.py"),
            "--task-group",
            "manipulation-expansion",
            "--num-envs",
            "4",
            "--steps",
            "4",
            "--seed",
            "11",
            "--json-out",
            str(json_out),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tasks"] == ["factory-peg-insert", "factory-gear-mesh"]
    assert payload["cpu_fallback_detected"] is False
    assert payload["benchmarks"][0]["task"] == "factory-peg-insert"
    assert payload["benchmarks"][0]["output_signature"]["final_peg_insert_depth_mean"] >= 0.0
    assert payload["benchmarks"][0]["output_signature"]["final_peg_variant_mean"] >= 0.0
    assert payload["benchmarks"][1]["task"] == "factory-gear-mesh"
    assert payload["benchmarks"][1]["output_signature"]["final_gear_mesh_insert_depth_mean"] >= 0.0
    assert payload["benchmarks"][1]["output_signature"]["final_gear_mesh_variant_mean"] >= 0.0


def test_factory_gear_mesh_single_task_benchmark_is_runnable(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[4]
    json_out = tmp_path / "factory-gear-mesh-benchmark.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmarks" / "mlx" / "benchmark_mac_tasks.py"),
            "--tasks",
            "factory-gear-mesh",
            "--num-envs",
            "4",
            "--steps",
            "4",
            "--seed",
            "13",
            "--json-out",
            str(json_out),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tasks"] == ["factory-gear-mesh"]
    assert payload["cpu_fallback_detected"] is False
    assert payload["benchmarks"][0]["task"] == "factory-gear-mesh"
    assert payload["benchmarks"][0]["output_signature"]["final_gear_mesh_insert_depth_mean"] >= 0.0
    assert payload["benchmarks"][0]["output_signature"]["final_gear_mesh_variant_mean"] >= 0.0
