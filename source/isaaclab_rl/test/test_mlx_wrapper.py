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
        "openarm-reach",
        "openarm-bi-reach",
        "ur10-reach",
        "ur10e-deploy-reach",
        "ur10e-gear-assembly-2f140",
        "ur10e-gear-assembly-2f85",
        "factory-peg-insert",
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
    assert get_mlx_task_spec("h1-flat").default_hidden_dim == 192
    assert get_mlx_task_spec("franka-reach").default_hidden_dim == 128
    assert get_mlx_task_spec("openarm-reach").default_hidden_dim == 128
    assert get_mlx_task_spec("openarm-reach").semantic_contract == "reduced-openarm-surrogate"
    assert get_mlx_task_spec("openarm-reach").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("openarm-bi-reach").default_hidden_dim == 160
    assert get_mlx_task_spec("openarm-bi-reach").semantic_contract == "reduced-openarm-bimanual-surrogate"
    assert get_mlx_task_spec("openarm-bi-reach").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("ur10-reach").default_hidden_dim == 128
    assert get_mlx_task_spec("ur10-reach").semantic_contract == "reduced-analytic-pose"
    assert get_mlx_task_spec("ur10-reach").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("ur10e-deploy-reach").default_hidden_dim == 128
    assert get_mlx_task_spec("ur10e-deploy-reach").semantic_contract == "reduced-analytic-pose"
    assert get_mlx_task_spec("ur10e-deploy-reach").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("ur10e-gear-assembly-2f140").default_hidden_dim == 128
    assert get_mlx_task_spec("ur10e-gear-assembly-2f140").semantic_contract == "reduced-analytic-assembly"
    assert get_mlx_task_spec("ur10e-gear-assembly-2f140").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("ur10e-gear-assembly-2f85").default_hidden_dim == 128
    assert get_mlx_task_spec("ur10e-gear-assembly-2f85").semantic_contract == "reduced-analytic-assembly"
    assert get_mlx_task_spec("ur10e-gear-assembly-2f85").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("factory-peg-insert").default_hidden_dim == 128
    assert get_mlx_task_spec("factory-peg-insert").semantic_contract == "reduced-analytic-peg-insert"
    assert get_mlx_task_spec("factory-peg-insert").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("ur10-long-suction-stack").default_hidden_dim == 128
    assert get_mlx_task_spec("ur10-long-suction-stack").semantic_contract == "reduced-analytic-suction-stack"
    assert get_mlx_task_spec("ur10-long-suction-stack").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("ur10-short-suction-stack").default_hidden_dim == 128
    assert get_mlx_task_spec("ur10-short-suction-stack").semantic_contract == "reduced-analytic-suction-stack"
    assert get_mlx_task_spec("ur10-short-suction-stack").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("franka-lift").default_hidden_dim == 128
    assert get_mlx_task_spec("openarm-lift").default_hidden_dim == 128
    assert get_mlx_task_spec("openarm-lift").semantic_contract == "reduced-openarm-surrogate"
    assert get_mlx_task_spec("openarm-lift").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("agibot-place-toy2box").default_hidden_dim == 128
    assert get_mlx_task_spec("agibot-place-toy2box").semantic_contract == "reduced-agibot-place-surrogate"
    assert get_mlx_task_spec("agibot-place-toy2box").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("agibot-place-upright-mug").default_hidden_dim == 128
    assert get_mlx_task_spec("agibot-place-upright-mug").semantic_contract == "reduced-agibot-place-surrogate"
    assert get_mlx_task_spec("agibot-place-upright-mug").upstream_alias_semantics_preserved is False
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
    assert get_mlx_task_spec("openarm-open-drawer").default_hidden_dim == 128
    assert get_mlx_task_spec("openarm-open-drawer").semantic_contract == "reduced-openarm-surrogate"
    assert get_mlx_task_spec("openarm-open-drawer").upstream_alias_semantics_preserved is False


def test_public_mlx_wrapper_normalizes_upstream_manipulation_alias_specs():
    """Upstream Franka task ids should resolve to the canonical public MLX task specs."""

    assert get_mlx_task_spec("Isaac-Reach-Franka-IK-Abs-v0") == get_mlx_task_spec("franka-reach")
    assert get_mlx_task_spec("Isaac-Reach-OpenArm-v0") == get_mlx_task_spec("openarm-reach")
    assert get_mlx_task_spec("Isaac-Reach-OpenArm-Bi-Play-v0") == get_mlx_task_spec("openarm-bi-reach")
    assert get_mlx_task_spec("Isaac-Reach-UR10-Play-v0") == get_mlx_task_spec("ur10-reach")
    assert get_mlx_task_spec("Isaac-Deploy-Reach-UR10e-Play-v0") == get_mlx_task_spec("ur10e-deploy-reach")
    assert get_mlx_task_spec("Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0") == get_mlx_task_spec(
        "ur10e-gear-assembly-2f140"
    )
    assert get_mlx_task_spec("Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0") == get_mlx_task_spec(
        "ur10e-gear-assembly-2f85"
    )
    assert get_mlx_task_spec("Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0").task == "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0"
    assert get_mlx_task_spec("Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0").semantic_contract == "reduced-no-ros-inference"
    assert get_mlx_task_spec("Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0").task == "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0"
    assert get_mlx_task_spec("Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0").semantic_contract == "reduced-no-ros-inference"
    assert get_mlx_task_spec("Isaac-Factory-PegInsert-Direct-v0").task == "Isaac-Factory-PegInsert-Direct-v0"
    assert get_mlx_task_spec("Isaac-Factory-PegInsert-Direct-v0").semantic_contract == "reduced-analytic-peg-insert"
    assert get_mlx_task_spec("Isaac-Factory-PegInsert-Direct-v0").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("Isaac-Lift-Cube-Franka-IK-Abs-v0") == get_mlx_task_spec("franka-lift")
    assert get_mlx_task_spec("Isaac-Lift-Cube-Franka-IK-Rel-Play-v0") == get_mlx_task_spec("franka-lift")
    assert get_mlx_task_spec("Isaac-Lift-Cube-OpenArm-Play-v0") == get_mlx_task_spec("openarm-lift")
    assert get_mlx_task_spec("Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0") == get_mlx_task_spec("franka-teddy-bear-lift")
    assert get_mlx_task_spec("Isaac-Stack-Cube-Instance-Randomize-Franka-Play-v0") == get_mlx_task_spec(
        "franka-stack-instance-randomize"
    )
    assert get_mlx_task_spec("Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0") == get_mlx_task_spec(
        "franka-stack-instance-randomize"
    )
    assert get_mlx_task_spec("Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0") == get_mlx_task_spec(
        "franka-stack-instance-randomize"
    )
    assert get_mlx_task_spec("Isaac-Stack-Cube-Franka-IK-Abs-Play-v0") == get_mlx_task_spec("franka-stack")
    assert get_mlx_task_spec("Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-Play-v0") == get_mlx_task_spec("franka-stack")
    assert get_mlx_task_spec("Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-Play-v0") == get_mlx_task_spec("franka-stack")
    assert get_mlx_task_spec("Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0") == get_mlx_task_spec("franka-stack-rgb")
    assert get_mlx_task_spec("Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-Play-v0") == get_mlx_task_spec(
        "franka-stack-rgb"
    )
    assert get_mlx_task_spec("Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0") == get_mlx_task_spec("franka-bin-stack")
    assert get_mlx_task_spec("Isaac-PickPlace-GR1T2-Abs-v0").task == "Isaac-PickPlace-GR1T2-Abs-v0"
    assert get_mlx_task_spec("Isaac-PickPlace-GR1T2-Abs-v0").semantic_contract == "reduced-pick-place-surrogate"
    assert get_mlx_task_spec("Isaac-PickPlace-GR1T2-Abs-v0").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0").task == "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0"
    assert get_mlx_task_spec("Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0").semantic_contract == "reduced-pick-place-surrogate"
    assert get_mlx_task_spec("Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0").upstream_alias_semantics_preserved is False
    assert get_mlx_task_spec("Isaac-Open-Drawer-Franka-IK-Rel-v0") == get_mlx_task_spec("franka-open-drawer")
    assert get_mlx_task_spec("Isaac-Open-Drawer-OpenArm-Play-v0") == get_mlx_task_spec("openarm-open-drawer")


@pytest.mark.parametrize(
    ("alias_task", "canonical_task", "semantic_contract", "note_fragment"),
    (
        ("Isaac-Deploy-Reach-UR10e-ROS-Inference-v0", "ur10e-deploy-reach", "reduced-no-ros-inference", "ros inference"),
        (
            "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0",
            "ur10e-gear-assembly-2f140",
            "reduced-no-ros-inference",
            "ros inference",
        ),
        (
            "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0",
            "ur10e-gear-assembly-2f85",
            "reduced-no-ros-inference",
            "ros inference",
        ),
        (
            "Isaac-Factory-PegInsert-Direct-v0",
            "factory-peg-insert",
            "reduced-analytic-peg-insert",
            "peg-insert",
        ),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0", "franka-stack", "reduced-no-blueprint", "blueprint"),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0", "franka-stack", "reduced-no-skillgen", "skill-generation"),
        (
            "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
            "franka-stack-rgb",
            "reduced-visuomotor-surrogate",
            "synthetic rgb",
        ),
        (
            "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
            "franka-stack-rgb",
            "reduced-no-cosmos",
            "cosmos",
        ),
        (
            "Isaac-PickPlace-GR1T2-Abs-v0",
            "franka-bin-stack",
            "reduced-pick-place-surrogate",
            "pick-place",
        ),
        (
            "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
            "franka-bin-stack",
            "reduced-pick-place-surrogate",
            "pick-place",
        ),
        (
            "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
            "franka-bin-stack",
            "reduced-pick-place-surrogate",
            "pick-place",
        ),
        (
            "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
            "agibot-place-toy2box",
            "reduced-agibot-place-surrogate",
            "agibot",
        ),
        (
            "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
            "agibot-place-upright-mug",
            "reduced-agibot-place-surrogate",
            "agibot",
        ),
        (
            "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
            "franka-bin-stack",
            "reduced-pick-place-surrogate",
            "pick-place",
        ),
        (
            "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
            "franka-bin-stack",
            "reduced-pick-place-surrogate",
            "pick-place",
        ),
    ),
)
def test_public_mlx_wrapper_exposes_reduced_alias_specs(
    alias_task: str,
    canonical_task: str,
    semantic_contract: str,
    note_fragment: str,
):
    """Reduced upstream aliases should carry explicit non-parity task specs."""

    alias_spec = get_mlx_task_spec(alias_task)
    canonical_spec = get_mlx_task_spec(canonical_task)

    assert alias_spec.task == alias_task
    assert alias_spec.trainable is canonical_spec.trainable
    assert alias_spec.default_hidden_dim == canonical_spec.default_hidden_dim
    assert alias_spec.default_action_std == canonical_spec.default_action_std
    assert alias_spec.semantic_contract == semantic_contract
    assert alias_spec.upstream_alias_semantics_preserved is False
    assert note_fragment in alias_spec.notes.lower()


def test_public_mlx_wrapper_normalizes_pick_place_aliases_to_bin_stack(tmp_path: Path):
    """Pick-place aliases should normalize through the public MLX wrapper to the reduced bin-stack slice."""

    checkpoint_path = tmp_path / "pick-place-policy.npz"

    train_payload = train_mlx_task(
        "Isaac-PickPlace-GR1T2-Abs-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=61,
    )
    eval_payload = evaluate_mlx_task(
        "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
        episodes=1,
        checkpoint=str(checkpoint_path),
        episode_length_s=0.5,
        max_steps=256,
        seed=63,
    )

    assert train_payload["task"] == "franka-bin-stack"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert train_payload["task_spec"]["semantic_contract"] == "reduced-pick-place-surrogate"
    assert train_payload["task_spec"]["upstream_alias_semantics_preserved"] is False
    assert eval_payload["task"] == "franka-bin-stack"
    assert eval_payload["episodes_completed"] == 1


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


def test_evaluate_manipulation_and_ur10_manual_slices_via_public_mlx_wrapper():
    """The public wrapper should expose manual evaluation for the trainable manipulation slices."""

    reach_payload = evaluate_mlx_task(
        "franka-reach",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=31,
    )
    openarm_reach_payload = evaluate_mlx_task(
        "openarm-reach",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=32,
    )
    openarm_bi_reach_payload = evaluate_mlx_task(
        "openarm-bi-reach",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=33,
    )
    ur10_payload = evaluate_mlx_task(
        "ur10-reach",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=34,
    )
    ur10e_payload = evaluate_mlx_task(
        "ur10e-deploy-reach",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=35,
    )
    openarm_lift_payload = evaluate_mlx_task(
        "openarm-lift",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=38,
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
    openarm_open_drawer_payload = evaluate_mlx_task(
        "openarm-open-drawer",
        num_envs=8,
        episodes=1,
        episode_length_s=0.5,
        max_steps=512,
        random_actions=False,
        seed=52,
    )

    assert reach_payload["task"] == "franka-reach"
    assert reach_payload["episodes_completed"] == 1
    assert openarm_reach_payload["task"] == "openarm-reach"
    assert openarm_reach_payload["episodes_completed"] == 1
    assert openarm_bi_reach_payload["task"] == "openarm-bi-reach"
    assert openarm_bi_reach_payload["episodes_completed"] == 1
    assert ur10_payload["task"] == "ur10-reach"
    assert ur10_payload["episodes_completed"] == 1
    assert ur10e_payload["task"] == "ur10e-deploy-reach"
    assert ur10e_payload["episodes_completed"] == 1
    assert lift_payload["task"] == "franka-lift"
    assert lift_payload["episodes_completed"] == 1
    assert openarm_lift_payload["task"] == "openarm-lift"
    assert openarm_lift_payload["episodes_completed"] == 1
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
    assert openarm_open_drawer_payload["task"] == "openarm-open-drawer"
    assert openarm_open_drawer_payload["episodes_completed"] == 1


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


def test_train_and_evaluate_openarm_reach_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for reduced OpenArm reach."""

    checkpoint_path = tmp_path / "openarm-reach-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "openarm-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=42,
    )
    eval_payload = evaluate_mlx_task(
        "openarm-reach",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=42,
    )

    assert train_payload["task"] == "openarm-reach"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "openarm-reach"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_openarm_bi_reach_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for reduced OpenArm bimanual reach."""

    checkpoint_path = tmp_path / "openarm-bi-reach-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "openarm-bi-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=48,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=43,
    )
    eval_payload = evaluate_mlx_task(
        "openarm-bi-reach",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=43,
    )

    assert train_payload["task"] == "openarm-bi-reach"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "openarm-bi-reach"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_ur10_reach_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for reduced UR10 reach."""

    checkpoint_path = tmp_path / "ur10-reach-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "ur10-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=44,
    )
    eval_payload = evaluate_mlx_task(
        "ur10-reach",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=44,
    )

    assert train_payload["task"] == "ur10-reach"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "ur10-reach"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


def test_train_and_evaluate_ur10e_deploy_reach_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for UR10e deploy-reach."""

    checkpoint_path = tmp_path / "ur10e-deploy-reach-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "ur10e-deploy-reach",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=42,
    )
    eval_payload = evaluate_mlx_task(
        "ur10e-deploy-reach",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=42,
    )

    assert train_payload["task"] == "ur10e-deploy-reach"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "ur10e-deploy-reach"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


@pytest.mark.parametrize(
    ("task", "seed"),
    (
        ("ur10-long-suction-stack", 45),
        ("ur10-short-suction-stack", 46),
    ),
)
def test_train_and_evaluate_ur10_suction_stack_via_public_mlx_wrapper(
    tmp_path: Path,
    task: str,
    seed: int,
):
    """The public wrapper should expose train/replay surfaces for the new UR10 stack slices."""

    checkpoint_path = tmp_path / f"{task}-wrapper-policy.npz"

    train_payload = train_mlx_task(
        task,
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=seed,
    )
    eval_payload = evaluate_mlx_task(
        task,
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=seed,
    )

    assert train_payload["task"] == task
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == task
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


@pytest.mark.parametrize(
    ("task", "seed"),
    (
        ("ur10e-gear-assembly-2f140", 46),
        ("ur10e-gear-assembly-2f85", 47),
    ),
)
def test_train_and_evaluate_ur10e_gear_assembly_via_public_mlx_wrapper(tmp_path: Path, task: str, seed: int):
    """The public wrapper should expose train/replay surfaces for reduced UR10e gear-assembly slices."""

    checkpoint_path = tmp_path / f"{task}-wrapper-policy.npz"

    train_payload = train_mlx_task(
        task,
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=seed,
    )
    eval_payload = evaluate_mlx_task(
        task,
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=seed,
    )

    assert train_payload["task"] == task
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == task
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


def test_train_and_evaluate_openarm_lift_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for reduced OpenArm lift."""

    checkpoint_path = tmp_path / "openarm-lift-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "openarm-lift",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=45,
    )
    eval_payload = evaluate_mlx_task(
        "openarm-lift",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=45,
    )

    assert train_payload["task"] == "openarm-lift"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "openarm-lift"
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert isinstance(eval_payload["completed"][0]["return"], float)


@pytest.mark.parametrize(
    ("task", "seed"),
    (
        ("agibot-place-toy2box", 145),
        ("agibot-place-upright-mug", 147),
    ),
)
def test_train_and_evaluate_agibot_place_via_public_mlx_wrapper(tmp_path: Path, task: str, seed: int):
    """The public wrapper should expose train/replay surfaces for reduced Agibot place slices."""

    checkpoint_path = tmp_path / f"{task}-wrapper-policy.npz"

    train_payload = train_mlx_task(
        task,
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=seed,
    )
    eval_payload = evaluate_mlx_task(
        task,
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=seed,
    )

    assert train_payload["task"] == task
    assert Path(train_payload["checkpoint_path"]).exists()
    assert train_payload["task_spec"]["semantic_contract"] == "reduced-agibot-place-surrogate"
    assert train_payload["task_spec"]["upstream_alias_semantics_preserved"] is False
    assert eval_payload["task"] == task
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert eval_payload["task_spec"]["semantic_contract"] == "reduced-agibot-place-surrogate"
    assert eval_payload["task_spec"]["upstream_alias_semantics_preserved"] is False


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


def test_public_mlx_wrapper_normalizes_factory_peg_insert_alias_to_canonical(tmp_path: Path):
    """The public wrapper should expose the reduced factory peg-insert slice end to end."""

    checkpoint_path = tmp_path / "factory_peg_insert_wrapper_policy.npz"

    train_payload = train_mlx_task(
        "Isaac-Factory-PegInsert-Direct-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=77,
    )
    eval_payload = evaluate_mlx_task(
        "Isaac-Factory-PegInsert-Direct-v0",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=77,
    )

    assert train_payload["task"] == "factory-peg-insert"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert train_payload["task_spec"]["semantic_contract"] == "reduced-analytic-peg-insert"
    assert train_payload["task_spec"]["upstream_alias_semantics_preserved"] is False
    assert eval_payload["task"] == "factory-peg-insert"
    assert eval_payload["episodes_completed"] == 1
    assert eval_payload["task_spec"]["semantic_contract"] == "reduced-analytic-peg-insert"
    assert eval_payload["task_spec"]["upstream_alias_semantics_preserved"] is False
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


def test_train_and_evaluate_openarm_open_drawer_via_public_mlx_wrapper(tmp_path: Path):
    """The public wrapper should expose a train/replay surface for reduced OpenArm open-drawer."""

    checkpoint_path = tmp_path / "openarm-open-drawer-wrapper-policy.npz"

    train_payload = train_mlx_task(
        "openarm-open-drawer",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=71,
    )
    eval_payload = evaluate_mlx_task(
        "openarm-open-drawer",
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=71,
    )

    assert train_payload["task"] == "openarm-open-drawer"
    assert Path(train_payload["checkpoint_path"]).exists()
    assert eval_payload["task"] == "openarm-open-drawer"
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

    ur10e_checkpoint = tmp_path / "ur10e-deploy-reach-alias-policy.npz"
    ur10e_train_payload = train_mlx_task(
        "Isaac-Deploy-Reach-UR10e-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(ur10e_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=70,
    )
    ur10e_eval_payload = evaluate_mlx_task(
        "Isaac-Deploy-Reach-UR10e-Play-v0",
        checkpoint=str(ur10e_checkpoint),
        episodes=1,
        episode_length_s=0.5,
        seed=70,
    )
    assert ur10e_train_payload["task"] == "ur10e-deploy-reach"
    assert Path(ur10e_train_payload["checkpoint_path"]).exists()
    assert ur10e_eval_payload["task"] == "ur10e-deploy-reach"
    assert ur10e_eval_payload["mode"] == "checkpoint"
    assert ur10e_eval_payload["episodes_completed"] == 1

    gear_2f140_checkpoint = tmp_path / "ur10e-gear-assembly-2f140-alias-policy.npz"
    gear_2f140_train_payload = train_mlx_task(
        "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(gear_2f140_checkpoint),
        eval_interval=1,
        episode_length_s=0.5,
        seed=170,
    )
    gear_2f140_eval_payload = evaluate_mlx_task(
        "Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0",
        checkpoint=str(gear_2f140_checkpoint),
        episodes=1,
        episode_length_s=0.5,
        seed=170,
    )
    assert gear_2f140_train_payload["task"] == "ur10e-gear-assembly-2f140"
    assert Path(gear_2f140_train_payload["checkpoint_path"]).exists()
    assert gear_2f140_eval_payload["task"] == "ur10e-gear-assembly-2f140"
    assert gear_2f140_eval_payload["mode"] == "checkpoint"
    assert gear_2f140_eval_payload["episodes_completed"] == 1
    assert gear_2f140_eval_payload["task_spec"]["semantic_contract"] == "reduced-analytic-assembly"
    assert gear_2f140_eval_payload["task_spec"]["upstream_alias_semantics_preserved"] is False

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


@pytest.mark.parametrize(
    ("alias_task", "canonical_task", "semantic_contract"),
    (
        ("Isaac-Deploy-Reach-UR10e-ROS-Inference-v0", "ur10e-deploy-reach", "reduced-no-ros-inference"),
        ("Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0", "ur10e-gear-assembly-2f140", "reduced-no-ros-inference"),
        ("Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0", "ur10e-gear-assembly-2f85", "reduced-no-ros-inference"),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Play-v0", "franka-stack", "reduced-analytic-stack"),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0", "franka-stack", "reduced-no-blueprint"),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0", "franka-stack", "reduced-no-skillgen"),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0", "franka-stack-rgb", "reduced-visuomotor-surrogate"),
        ("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0", "franka-stack-rgb", "reduced-no-cosmos"),
    ),
)
def test_train_and_evaluate_reduced_upstream_aliases_via_public_mlx_wrapper(
    tmp_path: Path,
    alias_task: str,
    canonical_task: str,
    semantic_contract: str,
):
    """Reduced upstream aliases should train and replay while preserving canonical task ids in payloads."""

    checkpoint_path = tmp_path / f"{canonical_task}-{semantic_contract}-wrapper-policy.npz"

    train_payload = train_mlx_task(
        alias_task,
        num_envs=8,
        updates=1,
        rollout_steps=8,
        epochs_per_update=1,
        hidden_dim=32,
        checkpoint=str(checkpoint_path),
        eval_interval=1,
        episode_length_s=0.5,
        seed=79,
    )
    eval_payload = evaluate_mlx_task(
        alias_task,
        checkpoint=str(checkpoint_path),
        episodes=1,
        episode_length_s=0.5,
        seed=79,
    )

    assert train_payload["task"] == canonical_task
    assert Path(train_payload["checkpoint_path"]).exists()
    assert train_payload["task_spec"]["semantic_contract"] == semantic_contract
    assert train_payload["task_spec"]["upstream_alias_semantics_preserved"] is False
    assert eval_payload["task"] == canonical_task
    assert eval_payload["mode"] == "checkpoint"
    assert eval_payload["episodes_completed"] == 1
    assert eval_payload["task_spec"]["semantic_contract"] == semantic_contract
    assert eval_payload["task_spec"]["upstream_alias_semantics_preserved"] is False


def test_public_mlx_wrapper_rejects_non_trainable_tasks():
    """The wrapper should fail explicitly when training is requested for eval-only tasks."""
    with pytest.raises(ValueError, match="does not expose an MLX training surface"):
        train_mlx_task("quadcopter", updates=1, rollout_steps=8, epochs_per_update=1)


def test_public_mlx_wrapper_rejects_unknown_tasks():
    """The wrapper should fail explicitly when an unsupported task id is requested."""
    with pytest.raises(ValueError, match="Unsupported MLX task"):
        get_mlx_task_spec("shadow-hand-vision")

    with pytest.raises(ValueError, match="Unsupported MLX task"):
        get_mlx_task_spec("Isaac-Tracking-LocoManip-Digit-v0")

    with pytest.raises(ValueError, match="Unsupported MLX evaluation task"):
        evaluate_mlx_task("shadow-hand-vision")

    with pytest.raises(ValueError, match="Unsupported MLX evaluation task"):
        evaluate_mlx_task("Isaac-Tracking-LocoManip-Digit-v0")


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
