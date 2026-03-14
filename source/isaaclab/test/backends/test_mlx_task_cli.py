# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the shared MLX/mac-sim task CLIs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from isaaclab.backends.kernel_inventory import CURRENT_MAC_NATIVE_TASKS
from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()


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

    assert module.EVAL_TASKS == CURRENT_MAC_NATIVE_TASKS
    assert module.TRAIN_TASKS == ("cartpole", "anymal-c-flat", "h1-flat")


def test_shared_task_cli_trains_first_locomotion_task(tmp_path: Path):
    module = _load_task_support_module()
    checkpoint_path = tmp_path / "anymal_c_flat_policy.npz"

    payload = module._train_task(
        "anymal-c-flat",
        argparse.Namespace(
            num_envs=8,
            updates=1,
            rollout_steps=8,
            epochs_per_update=1,
            learning_rate=3e-4,
            hidden_dim=32,
            action_std=None,
            checkpoint=str(checkpoint_path),
            resume_from=None,
            eval_interval=1,
            episode_length_s=0.5,
            seed=13,
            json_out=None,
        ),
    )

    assert payload["task"] == "anymal-c-flat"
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metadata_path"]).exists()
    assert payload["completed_episodes"] >= 0


def test_shared_task_cli_evaluates_h1_manual_slice():
    module = _load_task_support_module()

    payload = module._evaluate_task(
        "h1-flat",
        argparse.Namespace(
            num_envs=8,
            episodes=1,
            seed=17,
            episode_length_s=0.5,
            max_steps=256,
            checkpoint=None,
            hidden_dim=None,
            random_actions=False,
            cart_action=0.0,
            pendulum_action=0.0,
            thrust_action=0.2,
            roll_action=0.0,
            pitch_action=0.0,
            yaw_action=0.0,
            json_out=None,
        ),
    )

    assert payload["task"] == "h1-flat"
    assert payload["mode"] == "manual"
    assert payload["episodes_requested"] == 1
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0
