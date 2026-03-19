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
        "cartpole-rgb-camera",
        "cartpole-depth-camera",
        "cart-double-pendulum",
        "quadcopter",
        "anymal-c-flat",
        "anymal-c-rough",
        "h1-flat",
        "h1-rough",
        "franka-reach",
        "franka-lift",
        "franka-teddy-bear-lift",
        "franka-stack-instance-randomize",
        "franka-stack",
        "franka-stack-rgb",
        "franka-bin-stack",
        "franka-cabinet",
        "franka-open-drawer",
    )
    assert list_trainable_mlx_tasks() == (
        "cartpole",
        "anymal-c-flat",
        "anymal-c-rough",
        "h1-flat",
        "h1-rough",
        "franka-reach",
        "franka-lift",
        "franka-teddy-bear-lift",
        "franka-stack-instance-randomize",
        "franka-stack",
        "franka-stack-rgb",
        "franka-bin-stack",
        "franka-cabinet",
        "franka-open-drawer",
    )
    assert get_mlx_task_spec("h1-flat").default_hidden_dim == 192
    assert get_mlx_task_spec("franka-reach").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-lift").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-teddy-bear-lift").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-stack-instance-randomize").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-stack").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-stack-rgb").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-bin-stack").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-bin-stack").semantic_contract == "reduced-no-mimic"
    assert get_mlx_task_spec("franka-bin-stack").upstream_alias_semantics_preserved is False
    assert "mimic" in get_mlx_task_spec("franka-bin-stack").notes.lower()
    assert get_mlx_task_spec("franka-cabinet").default_hidden_dim == 128
    assert get_mlx_task_spec("franka-open-drawer").default_hidden_dim == 128


def test_public_mlx_wrapper_normalizes_upstream_manipulation_alias_specs():
    """Upstream Franka task ids should resolve to the canonical public MLX task specs."""

    assert get_mlx_task_spec("Isaac-Reach-Franka-IK-Abs-v0") == get_mlx_task_spec("franka-reach")
    assert get_mlx_task_spec("Isaac-Lift-Cube-Franka-IK-Abs-v0") == get_mlx_task_spec("franka-lift")
    assert get_mlx_task_spec("Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0") == get_mlx_task_spec("franka-teddy-bear-lift")
    assert get_mlx_task_spec("Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0") == get_mlx_task_spec(
        "franka-stack-instance-randomize"
    )
    assert get_mlx_task_spec("Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0") == get_mlx_task_spec("franka-stack-rgb")
    assert get_mlx_task_spec("Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0") == get_mlx_task_spec("franka-bin-stack")
    assert get_mlx_task_spec("Isaac-Open-Drawer-Franka-IK-Rel-v0") == get_mlx_task_spec("franka-open-drawer")


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


def test_train_and_evaluate_anymal_rough_via_public_mlx_wrapper(tmp_path: Path):
    """The MLX wrapper should expose checkpoint replay for the rough ANYmal-C slice."""

    checkpoint_path = tmp_path / "anymal_rough_wrapper_policy.npz"

    train_payload = train_mlx_task(
        "anymal-c-rough",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=23,
    )
    eval_payload = evaluate_mlx_task(
        "anymal-c-rough",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=23,
    )

    assert train_payload["task"] == "anymal-c-rough"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "anymal-c-rough"
    assert eval_payload["mode"] == "checkpoint"
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


def test_evaluate_h1_rough_manual_via_public_mlx_wrapper():
    """The public wrapper should expose manual evaluation for the rough H1 slice."""
    payload = evaluate_mlx_task(
        "h1-rough",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
        seed=33,
    )

    assert payload["task"] == "h1-rough"
    assert payload["mode"] == "manual"
    assert payload["episodes_completed"] == 1
    assert payload["completed"][0]["length"] > 0


def test_train_and_evaluate_h1_rough_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose checkpoint replay for the rough H1 slice."""

    checkpoint_path = tmp_path / "h1_rough_wrapper_policy.npz"

    train_payload = train_mlx_task(
        "h1-rough",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=41,
    )
    eval_payload = evaluate_mlx_task(
        "h1-rough",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=41,
    )

    assert train_payload["task"] == "h1-rough"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "h1-rough"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_evaluate_cartpole_camera_manual_via_public_mlx_wrapper():
    """The public wrapper should expose the synthetic camera cartpole slices."""

    rgb_payload = evaluate_mlx_task(
        "cartpole-rgb-camera",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
        seed=35,
    )
    depth_payload = evaluate_mlx_task(
        "cartpole-depth-camera",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=256,
        random_actions=False,
        seed=37,
    )

    assert rgb_payload["task"] == "cartpole-rgb-camera"
    assert rgb_payload["mode"] == "manual"
    assert rgb_payload["episodes_completed"] == 1
    assert depth_payload["task"] == "cartpole-depth-camera"
    assert depth_payload["mode"] == "manual"
    assert depth_payload["episodes_completed"] == 1


def test_evaluate_franka_reach_lift_stack_family_cabinet_and_open_drawer_manual_via_public_mlx_wrapper():
    """The public wrapper should expose manual evaluation for the current trainable manipulation slices."""

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
    stack_instance_payload = evaluate_mlx_task(
        "franka-stack-instance-randomize",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=45,
    )
    stack_payload = evaluate_mlx_task(
        "franka-stack",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=47,
    )
    stack_rgb_payload = evaluate_mlx_task(
        "franka-stack-rgb",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=48,
    )
    bin_stack_payload = evaluate_mlx_task(
        "franka-bin-stack",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=49,
    )
    cabinet_payload = evaluate_mlx_task(
        "franka-cabinet",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=50,
    )
    open_drawer_payload = evaluate_mlx_task(
        "franka-open-drawer",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=51,
    )

    assert reach_payload["task"] == "franka-reach"
    assert reach_payload["episodes_completed"] == 1
    assert lift_payload["task"] == "franka-lift"
    assert lift_payload["episodes_completed"] == 1
    assert stack_instance_payload["task"] == "franka-stack-instance-randomize"
    assert stack_instance_payload["episodes_completed"] == 1
    assert stack_payload["task"] == "franka-stack"
    assert stack_payload["episodes_completed"] == 1
    assert stack_rgb_payload["task"] == "franka-stack-rgb"
    assert stack_rgb_payload["episodes_completed"] == 1
    assert bin_stack_payload["task"] == "franka-bin-stack"
    assert bin_stack_payload["episodes_completed"] == 1
    assert cabinet_payload["task"] == "franka-cabinet"
    assert cabinet_payload["episodes_completed"] == 1
    assert open_drawer_payload["task"] == "franka-open-drawer"
    assert open_drawer_payload["episodes_completed"] == 1


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


def test_train_and_evaluate_franka_lift_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the lift manipulation task."""

    checkpoint_path = tmp_path / "franka-lift-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-lift",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=43,
    )
    eval_payload = evaluate_mlx_task(
        "franka-lift",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=43,
    )

    assert train_payload["task"] == "franka-lift"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-lift"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_teddy_bear_lift_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the teddy-bear lift slice."""

    checkpoint_path = tmp_path / "franka-teddy-bear-lift-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-teddy-bear-lift",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=47,
    )
    eval_payload = evaluate_mlx_task(
        "franka-teddy-bear-lift",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=47,
    )

    assert train_payload["task"] == "franka-teddy-bear-lift"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-teddy-bear-lift"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_stack_instance_randomize_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the instance-randomized stack slice."""

    checkpoint_path = tmp_path / "franka-stack-instance-randomize-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-stack-instance-randomize",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=49,
    )
    eval_payload = evaluate_mlx_task(
        "franka-stack-instance-randomize",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=49,
    )

    assert train_payload["task"] == "franka-stack-instance-randomize"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-stack-instance-randomize"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_stack_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the stack manipulation task."""

    checkpoint_path = tmp_path / "franka-stack-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-stack",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=53,
    )
    eval_payload = evaluate_mlx_task(
        "franka-stack",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=53,
    )

    assert train_payload["task"] == "franka-stack"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-stack"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_stack_rgb_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the three-cube stack task."""

    checkpoint_path = tmp_path / "franka-stack-rgb-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-stack-rgb",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=57,
    )
    eval_payload = evaluate_mlx_task(
        "franka-stack-rgb",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=57,
    )

    assert train_payload["task"] == "franka-stack-rgb"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-stack-rgb"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_bin_stack_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the bin-anchored stack task."""

    checkpoint_path = tmp_path / "franka-bin-stack-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-bin-stack",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=58,
    )
    eval_payload = evaluate_mlx_task(
        "franka-bin-stack",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=58,
    )

    assert train_payload["task"] == "franka-bin-stack"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-bin-stack"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_cabinet_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the cabinet manipulation task."""

    checkpoint_path = tmp_path / "franka-cabinet-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-cabinet",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=59,
    )
    eval_payload = evaluate_mlx_task(
        "franka-cabinet",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=59,
    )

    assert train_payload["task"] == "franka-cabinet"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-cabinet"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_franka_open_drawer_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for the open-drawer task."""

    checkpoint_path = tmp_path / "franka-open-drawer-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "franka-open-drawer",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=61,
    )
    eval_payload = evaluate_mlx_task(
        "franka-open-drawer",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=61,
    )

    assert train_payload["task"] == "franka-open-drawer"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "franka-open-drawer"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_upstream_manipulation_aliases_via_public_mlx_wrapper(tmp_path: Path):
    """Upstream-compatible Franka task ids should route through the canonical MLX manipulation slices."""

    reach_checkpoint = tmp_path / "franka-reach-alias-policy.npz"
    reach_train_payload = train_mlx_task(
        "Isaac-Reach-Franka-IK-Abs-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(reach_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=67,
    )
    assert reach_train_payload["task"] == "franka-reach"
    assert Path(reach_train_payload["checkpoint_path"]).exists()

    lift_checkpoint = tmp_path / "franka-lift-alias-policy.npz"
    lift_train_payload = train_mlx_task(
        "Isaac-Lift-Cube-Franka-IK-Rel-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(lift_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=69,
    )
    lift_eval_payload = evaluate_mlx_task(
        "Isaac-Lift-Cube-Franka-IK-Rel-v0",
        checkpoint=str(lift_checkpoint),
        episodes=1,
        episode_length_s=0.5,
        seed=69,
    )
    assert lift_train_payload["task"] == "franka-lift"
    assert Path(lift_train_payload["checkpoint_path"]).exists()
    assert lift_eval_payload["task"] == "franka-lift"
    assert lift_eval_payload["mode"] == "checkpoint"
    assert lift_eval_payload["episodes_completed"] == 1

    stack_rgb_payload = evaluate_mlx_task(
        "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=71,
    )
    assert stack_rgb_payload["task"] == "franka-stack-rgb"
    assert stack_rgb_payload["mode"] == "manual"
    assert stack_rgb_payload["episodes_completed"] == 1

    bin_stack_checkpoint = tmp_path / "franka-bin-stack-alias-policy.npz"
    bin_stack_train_payload = train_mlx_task(
        "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(bin_stack_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=72,
    )
    bin_stack_eval_payload = evaluate_mlx_task(
        "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
        checkpoint=str(bin_stack_checkpoint),
        episodes=1,
        episode_length_s=0.5,
        seed=72,
    )
    assert bin_stack_train_payload["task"] == "franka-bin-stack"
    assert Path(bin_stack_train_payload["checkpoint_path"]).exists()
    assert bin_stack_eval_payload["task"] == "franka-bin-stack"
    assert bin_stack_eval_payload["mode"] == "checkpoint"
    assert bin_stack_eval_payload["episodes_completed"] == 1

    open_drawer_checkpoint = tmp_path / "franka-open-drawer-alias-policy.npz"
    open_drawer_train_payload = train_mlx_task(
        "Isaac-Open-Drawer-Franka-IK-Rel-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(open_drawer_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=73,
    )
    open_drawer_eval_payload = evaluate_mlx_task(
        "Isaac-Open-Drawer-Franka-IK-Rel-v0",
        checkpoint=str(open_drawer_checkpoint),
        episodes=1,
        episode_length_s=0.5,
        seed=73,
    )
    assert open_drawer_train_payload["task"] == "franka-open-drawer"
    assert Path(open_drawer_train_payload["checkpoint_path"]).exists()
    assert open_drawer_eval_payload["task"] == "franka-open-drawer"
    assert open_drawer_eval_payload["mode"] == "checkpoint"
    assert open_drawer_eval_payload["episodes_completed"] == 1


def test_public_mlx_wrapper_rejects_non_trainable_tasks():
    """The wrapper should fail explicitly when training is requested for eval-only tasks."""
    with pytest.raises(ValueError, match="does not expose an MLX training surface"):
        train_mlx_task("quadcopter", updates=1, rollout_steps=8, epochs_per_update=1)


def test_public_mlx_wrapper_rejects_unknown_tasks():
    """The wrapper should fail explicitly when an unsupported task id is requested."""
    with pytest.raises(ValueError, match="Unsupported MLX task"):
        get_mlx_task_spec("shadow-hand-vision")

    with pytest.raises(ValueError, match="Unsupported MLX task"):
        get_mlx_task_spec("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0")

    with pytest.raises(ValueError, match="Unsupported MLX evaluation task"):
        evaluate_mlx_task("shadow-hand-vision")

    with pytest.raises(ValueError, match="Unsupported MLX evaluation task"):
        evaluate_mlx_task("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0")


def test_public_mlx_wrapper_rejects_checkpoint_for_eval_only_tasks():
    """Eval-only tasks should fail explicitly instead of silently ignoring checkpoints."""
    with pytest.raises(ValueError, match="does not expose checkpoint replay"):
        evaluate_mlx_task("quadcopter", checkpoint="logs/mlx/quadcopter_policy.npz")

    with pytest.raises(ValueError, match="does not expose checkpoint replay"):
        evaluate_mlx_task("cart-double-pendulum", checkpoint="logs/mlx/cart_double_policy.npz")

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
