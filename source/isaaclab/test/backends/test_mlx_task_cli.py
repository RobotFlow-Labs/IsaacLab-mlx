# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the shared MLX/mac-sim task CLIs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    assert list_trainable_mlx_tasks() == ("cartpole", "anymal-c-flat", "h1-flat", "franka-reach", "franka-lift", "franka-stack")


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
