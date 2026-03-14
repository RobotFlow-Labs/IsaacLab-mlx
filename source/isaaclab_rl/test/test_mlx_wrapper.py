# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the public MLX-native RL wrapper surface."""

from __future__ import annotations

from pathlib import Path
import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()

from isaaclab_rl.mlx import (  # noqa: E402
    evaluate_mlx_task,
    get_mlx_task_spec,
    list_mlx_tasks,
    list_trainable_mlx_tasks,
    train_mlx_task,
)


def test_public_mlx_task_lists_are_stable():
    """The MLX wrapper should publish the supported task ids clearly."""
    assert list_mlx_tasks() == (
        "cartpole",
        "cart-double-pendulum",
        "quadcopter",
        "anymal-c-flat",
        "anymal-c-rough",
        "h1-flat",
        "franka-reach",
        "franka-lift",
    )
    assert list_trainable_mlx_tasks() == ("cartpole", "anymal-c-flat", "h1-flat", "franka-reach")
    assert get_mlx_task_spec("h1-flat").default_hidden_dim == 192
    assert get_mlx_task_spec("franka-reach").default_hidden_dim == 128


def test_train_and_evaluate_anymal_via_public_mlx_wrapper(tmp_path: Path):
    """The MLX wrapper should provide a stable train/evaluate surface for locomotion tasks."""
    checkpoint_path = tmp_path / "anymal-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "anymal-c-flat",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=19,
    )
    eval_payload = evaluate_mlx_task(
        "anymal-c-flat",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=19,
    )

    assert train_payload["task"] == "anymal-c-flat"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_cartpole_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should cover the first trainable control task as well."""
    checkpoint_path = tmp_path / "cartpole-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "cartpole",
        num_envs=16,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        seed=23,
    )
    eval_payload = evaluate_mlx_task(
        "cartpole",
        checkpoint=str(checkpoint_path),
        episodes=1,
        seed=23,
    )

    assert train_payload["task"] == "cartpole"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "cartpole"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_evaluate_h1_manual_via_public_mlx_wrapper():
    """The public wrapper should expose manual evaluation for trainable locomotion tasks."""
    payload = evaluate_mlx_task(
        "h1-flat",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
        seed=29,
    )

    assert payload["task"] == "h1-flat"
    assert payload["mode"] == "manual"
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_evaluate_franka_reach_and_lift_manual_via_public_mlx_wrapper():
    """The public wrapper should expose manual evaluation for the first manipulation slices."""

    reach_payload = evaluate_mlx_task(
        "franka-reach",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=31,
    )
    lift_payload = evaluate_mlx_task(
        "franka-lift",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=37,
    )

    assert reach_payload["task"] == "franka-reach"
    assert reach_payload["episodes_completed"] == 1
    assert lift_payload["task"] == "franka-lift"
    assert lift_payload["episodes_completed"] == 1


def test_train_and_evaluate_franka_reach_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the first manipulation task."""

    checkpoint_path = tmp_path / "franka-reach-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=41,
    )
    eval_payload = evaluate_mlx_task(
        "franka-reach",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=41,
    )

    assert train_payload["task"] == "franka-reach"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-reach"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_public_mlx_wrapper_rejects_non_trainable_tasks():
    """The wrapper should fail explicitly when training is requested for eval-only tasks."""
    with pytest.raises(ValueError, match="does not expose an MLX training surface"):
        train_mlx_task("quadcopter", updates=1, rollout_steps=8, epochs_per_update=1)


def test_public_mlx_wrapper_rejects_unknown_tasks():
    """The wrapper should fail explicitly when an unsupported task id is requested."""
    with pytest.raises(ValueError, match="Unsupported MLX task"):
        get_mlx_task_spec("shadow-hand-vision")

    with pytest.raises(ValueError, match="Unsupported MLX evaluation task"):
        evaluate_mlx_task("shadow-hand-vision")


def test_public_mlx_wrapper_rejects_checkpoint_for_eval_only_tasks():
    """Eval-only tasks should fail explicitly instead of silently ignoring checkpoints."""
    with pytest.raises(ValueError, match="does not expose checkpoint replay"):
        evaluate_mlx_task("quadcopter", checkpoint="logs/mlx/quadcopter_policy.npz")

    with pytest.raises(ValueError, match="does not expose checkpoint replay"):
        evaluate_mlx_task("cart-double-pendulum", checkpoint="logs/mlx/cart_double_policy.npz")

    with pytest.raises(ValueError, match="does not expose checkpoint replay"):
        evaluate_mlx_task("franka-lift", checkpoint="logs/mlx/franka_lift_policy.npz")


def test_public_mlx_wrapper_honors_short_episode_length_for_cart_double_pendulum():
    """The shared wrapper should pass episode_length_s through to eval-only task configs."""
    payload = evaluate_mlx_task(
        "cart-double-pendulum",
        num_envs=8,
        episodes=1,
        episode_length_s=0.1,
        max_steps=64,
        random_actions=False,
        seed=37,
    )

    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] <= 8
