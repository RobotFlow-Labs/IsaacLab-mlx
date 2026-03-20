# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable manifest for the public MLX/mac task surface."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

TaskFamily = Literal["control", "vision", "locomotion", "manipulation"]


@dataclass(frozen=True)
class MacNativeTaskSpec:
    """Stable metadata for a public mac-native MLX task slice."""

    key: str
    upstream_task_id: str
    family: TaskFamily
    benchmark_groups: tuple[str, ...]
    sensor_contract: tuple[str, ...]
    trainable: bool
    default_checkpoint: str | None
    default_hidden_dim: int | None
    default_action_std: float | None = None
    semantic_contract: str = "aligned"
    upstream_alias_semantics_preserved: bool = True
    notes: str = ""

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe task payload."""

        return asdict(self)


MAC_NATIVE_TASK_SPECS: tuple[MacNativeTaskSpec, ...] = (
    MacNativeTaskSpec(
        key="cartpole",
        upstream_task_id="Isaac-Cartpole-Direct-v0",
        family="control",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/cartpole_policy.npz",
        default_hidden_dim=128,
        notes="Discrete cartpole PPO slice.",
    ),
    MacNativeTaskSpec(
        key="cartpole-rgb-camera",
        upstream_task_id="Isaac-Cartpole-RGB-Camera-Direct-v0",
        family="vision",
        benchmark_groups=("sensor-mac-native",),
        sensor_contract=("synthetic-rgb",),
        trainable=False,
        default_checkpoint=None,
        default_hidden_dim=None,
        notes="Synthetic cartpole RGB camera slice with deterministic 100x100 observations.",
    ),
    MacNativeTaskSpec(
        key="cartpole-depth-camera",
        upstream_task_id="Isaac-Cartpole-Depth-Camera-Direct-v0",
        family="vision",
        benchmark_groups=("sensor-mac-native",),
        sensor_contract=("synthetic-depth",),
        trainable=False,
        default_checkpoint=None,
        default_hidden_dim=None,
        notes="Synthetic cartpole depth camera slice with deterministic 100x100 observations.",
    ),
    MacNativeTaskSpec(
        key="cart-double-pendulum",
        upstream_task_id="Isaac-Cart-Double-Pendulum-Direct-v0",
        family="control",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=False,
        default_checkpoint=None,
        default_hidden_dim=None,
        notes="MARL dict observation/action control slice.",
    ),
    MacNativeTaskSpec(
        key="quadcopter",
        upstream_task_id="Isaac-Quadcopter-Direct-v0",
        family="control",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=False,
        default_checkpoint=None,
        default_hidden_dim=None,
        notes="Root-state thrust and moment control slice.",
    ),
    MacNativeTaskSpec(
        key="anymal-c-flat",
        upstream_task_id="Isaac-Velocity-Flat-Anymal-C-Direct-v0",
        family="locomotion",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception", "optional-height-scan"),
        trainable=True,
        default_checkpoint="logs/mlx/anymal_c_flat_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.35,
        notes="Flat-terrain ANYmal-C locomotion with optional analytic height scan.",
    ),
    MacNativeTaskSpec(
        key="anymal-c-rough",
        upstream_task_id="Isaac-Velocity-Rough-Anymal-C-Direct-v0",
        family="locomotion",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception", "terrain-raycast"),
        trainable=True,
        default_checkpoint="logs/mlx/anymal_c_rough_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.35,
        notes="Procedural wave terrain plus analytic terrain raycasts.",
    ),
    MacNativeTaskSpec(
        key="h1-flat",
        upstream_task_id="Isaac-Velocity-Flat-H1-v0",
        family="locomotion",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception", "optional-height-scan"),
        trainable=True,
        default_checkpoint="logs/mlx/h1_flat_policy.npz",
        default_hidden_dim=192,
        default_action_std=0.28,
        notes="Flat-terrain H1 locomotion with optional analytic height scan.",
    ),
    MacNativeTaskSpec(
        key="h1-rough",
        upstream_task_id="Isaac-Velocity-Rough-H1-v0",
        family="locomotion",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception", "terrain-raycast"),
        trainable=True,
        default_checkpoint="logs/mlx/h1_rough_policy.npz",
        default_hidden_dim=192,
        default_action_std=0.28,
        notes="Procedural wave terrain plus analytic terrain raycasts.",
    ),
    MacNativeTaskSpec(
        key="franka-reach",
        upstream_task_id="Isaac-Reach-Franka-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_reach_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Analytic joint-space reach slice.",
    ),
    MacNativeTaskSpec(
        key="openarm-reach",
        upstream_task_id="Isaac-Reach-OpenArm-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/openarm_reach_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        semantic_contract="reduced-openarm-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic OpenArm unimanual reach slice preserving the reach workflow with a "
            "7-DoF surrogate rather than the exact OpenArm morphology."
        ),
    ),
    MacNativeTaskSpec(
        key="openarm-bi-reach",
        upstream_task_id="Isaac-Reach-OpenArm-Bi-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/openarm_bi_reach_policy.npz",
        default_hidden_dim=160,
        default_action_std=0.22,
        semantic_contract="reduced-openarm-bimanual-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic dual-arm OpenArm reach slice preserving the bimanual reach workflow "
            "with paired 7-DoF surrogates instead of the exact OpenArm body-frame stack."
        ),
    ),
    MacNativeTaskSpec(
        key="ur10-reach",
        upstream_task_id="Isaac-Reach-UR10-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/ur10_reach_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.2,
        semantic_contract="reduced-analytic-pose",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10 reach slice preserving the pose-tracking workflow with an "
            "analytic pose surrogate instead of the full UR10 controller stack."
        ),
    ),
    MacNativeTaskSpec(
        key="ur10e-deploy-reach",
        upstream_task_id="Isaac-Deploy-Reach-UR10e-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/ur10e_deploy_reach_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.2,
        semantic_contract="reduced-analytic-pose",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10e deploy-reach slice. The upstream task tracks a deployment-oriented "
            "pose command with real robot frame conventions; the mac-native slice preserves the joint-space "
            "reach workflow and pose-command observation shape with analytic pose tracking."
        ),
    ),
    MacNativeTaskSpec(
        key="ur10e-gear-assembly-2f140",
        upstream_task_id="Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/ur10e_gear_assembly_2f140_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.2,
        semantic_contract="reduced-analytic-assembly",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10e gear-assembly slice for the Robotiq 2F-140 gripper. "
            "The mac-native slice preserves the shaft-alignment and insertion workflow with "
            "scalar insertion progress instead of full factory contact dynamics and ROS deployment semantics."
        ),
    ),
    MacNativeTaskSpec(
        key="ur10e-gear-assembly-2f85",
        upstream_task_id="Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/ur10e_gear_assembly_2f85_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.2,
        semantic_contract="reduced-analytic-assembly",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10e gear-assembly slice for the Robotiq 2F-85 gripper. "
            "The mac-native slice preserves the shaft-alignment and insertion workflow with "
            "scalar insertion progress instead of full factory contact dynamics and ROS deployment semantics."
        ),
    ),
    MacNativeTaskSpec(
        key="ur10-long-suction-stack",
        upstream_task_id="Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/ur10_long_suction_stack_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.2,
        semantic_contract="reduced-analytic-suction-stack",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10 long-suction three-cube stack slice preserving the stack workflow "
            "with analytic pose tracking and suction-state surrogates instead of the upstream "
            "CPU-only surface-gripper stack."
        ),
    ),
    MacNativeTaskSpec(
        key="ur10-short-suction-stack",
        upstream_task_id="Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/ur10_short_suction_stack_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.2,
        semantic_contract="reduced-analytic-suction-stack",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10 short-suction three-cube stack slice preserving the stack workflow "
            "with analytic pose tracking and suction-state surrogates instead of the upstream "
            "CPU-only surface-gripper stack."
        ),
    ),
    MacNativeTaskSpec(
        key="franka-lift",
        upstream_task_id="Isaac-Lift-Cube-Franka-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_lift_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Analytic cube lift slice with lightweight grasp logic.",
    ),
    MacNativeTaskSpec(
        key="openarm-lift",
        upstream_task_id="Isaac-Lift-Cube-OpenArm-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/openarm_lift_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.22,
        semantic_contract="reduced-openarm-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic OpenArm cube-lift slice preserving the lift workflow with lightweight "
            "grasp logic rather than the exact OpenArm grasp geometry."
        ),
    ),
    MacNativeTaskSpec(
        key="franka-teddy-bear-lift",
        upstream_task_id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_teddy_bear_lift_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Reduced plush-object lift slice mapped onto the analytic lift substrate.",
    ),
    MacNativeTaskSpec(
        key="franka-stack-instance-randomize",
        upstream_task_id="Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_stack_instance_randomize_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Reduced instance-randomized two-cube stack slice with explicit variant-id observations.",
    ),
    MacNativeTaskSpec(
        key="franka-stack",
        upstream_task_id="Isaac-Stack-Cube-Franka-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_stack_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Analytic two-cube stack slice.",
    ),
    MacNativeTaskSpec(
        key="franka-stack-rgb",
        upstream_task_id="Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("synthetic-rgb", "proprioception"),
        trainable=True,
        default_checkpoint="logs/mlx/franka_stack_rgb_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Analytic three-cube sequential stack slice.",
    ),
    MacNativeTaskSpec(
        key="franka-bin-stack",
        upstream_task_id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_bin_stack_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        semantic_contract="reduced-no-mimic",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced bin-anchored three-cube stack slice. The upstream task id includes Mimic "
            "semantics, but the mac-native slice is a reduced PPO-style contract without imitation "
            "behavior. The appended bin-anchor observation tail explicitly mirrors the anchored support cube."
        ),
    ),
    MacNativeTaskSpec(
        key="franka-cabinet",
        upstream_task_id="Isaac-Franka-Cabinet-Direct-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_cabinet_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Reduced drawer-handle workflow keyed to the direct cabinet task.",
    ),
    MacNativeTaskSpec(
        key="franka-open-drawer",
        upstream_task_id="Isaac-Open-Drawer-Franka-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/franka_open_drawer_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.25,
        notes="Manager-style open-drawer task mapped onto the reduced analytic drawer substrate.",
    ),
    MacNativeTaskSpec(
        key="openarm-open-drawer",
        upstream_task_id="Isaac-Open-Drawer-OpenArm-v0",
        family="manipulation",
        benchmark_groups=("current-mac-native",),
        sensor_contract=("proprioception",),
        trainable=True,
        default_checkpoint="logs/mlx/openarm_open_drawer_policy.npz",
        default_hidden_dim=128,
        default_action_std=0.22,
        semantic_contract="reduced-openarm-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic OpenArm open-drawer slice preserving the drawer workflow with "
            "lightweight grasp logic rather than the exact cabinet scene."
        ),
    ),
)

BENCHMARK_ONLY_TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "sensor-mac-native": (
        "cartpole-rgb-camera",
        "cartpole-depth-camera",
        "anymal-c-flat-height-scan",
        "h1-flat-height-scan",
    ),
    "training-mac-native": ("train-cartpole",),
}


def mac_native_task_specs() -> tuple[MacNativeTaskSpec, ...]:
    """Return the stable ordered public MLX/mac task specs."""

    return MAC_NATIVE_TASK_SPECS


def mac_native_task_spec_map() -> dict[str, MacNativeTaskSpec]:
    """Return the task manifest keyed by public MLX/mac task id."""

    return {task.key: task for task in MAC_NATIVE_TASK_SPECS}


def list_public_mlx_tasks() -> tuple[str, ...]:
    """Return the stable public MLX/mac task ids."""

    return tuple(task.key for task in MAC_NATIVE_TASK_SPECS)


def list_current_mac_native_tasks() -> tuple[str, ...]:
    """Return the stable benchmarked public MLX/mac task group."""

    return tuple(task.key for task in MAC_NATIVE_TASK_SPECS if "current-mac-native" in task.benchmark_groups)


def list_trainable_mlx_tasks() -> tuple[str, ...]:
    """Return the public MLX/mac tasks that expose a train surface."""

    return tuple(task.key for task in MAC_NATIVE_TASK_SPECS if task.trainable)


def public_benchmark_task_groups() -> dict[str, tuple[str, ...]]:
    """Return benchmark groups derived only from the public typed task manifest."""
    groups: dict[str, list[str]] = {}
    for task in MAC_NATIVE_TASK_SPECS:
        for group in task.benchmark_groups:
            groups.setdefault(group, []).append(task.key)
    return {group: tuple(tasks) for group, tasks in groups.items()}


def benchmark_task_groups() -> dict[str, tuple[str, ...]]:
    """Return the stable benchmark task groups exposed on the public MLX/mac path."""

    groups = {group: list(tasks) for group, tasks in public_benchmark_task_groups().items()}
    for group, tasks in BENCHMARK_ONLY_TASK_GROUPS.items():
        groups.setdefault(group, [])
        groups[group].extend(tasks)
        groups[group] = list(dict.fromkeys(groups[group]))
    payload = {group: tuple(tasks) for group, tasks in groups.items()}
    payload["full"] = tuple(
        dict.fromkeys(
            list_public_mlx_tasks()
            + BENCHMARK_ONLY_TASK_GROUPS["sensor-mac-native"]
            + BENCHMARK_ONLY_TASK_GROUPS["training-mac-native"]
        )
    )
    return payload


def supported_task_surface_summary() -> dict[str, Any]:
    """Return a stable JSON-safe summary of the public MLX/mac task surface."""

    families = Counter(task.family for task in MAC_NATIVE_TASK_SPECS)
    sensor_contracts = Counter(contract for task in MAC_NATIVE_TASK_SPECS for contract in task.sensor_contract)
    public_benchmark_groups_payload = {
        group: list(tasks)
        for group, tasks in public_benchmark_task_groups().items()
    }
    benchmark_groups_payload = {
        group: list(tasks)
        for group, tasks in benchmark_task_groups().items()
    }
    return {
        "public_task_count": len(MAC_NATIVE_TASK_SPECS),
        "current_mac_native_count": len(list_current_mac_native_tasks()),
        "trainable_task_count": len(list_trainable_mlx_tasks()),
        "benchmark_only_group_count": len(BENCHMARK_ONLY_TASK_GROUPS),
        "families": dict(sorted(families.items())),
        "sensor_contracts": dict(sorted(sensor_contracts.items())),
        "public_benchmark_groups": public_benchmark_groups_payload,
        "benchmark_task_groups": benchmark_groups_payload,
        "benchmark_only_groups": {group: list(tasks) for group, tasks in BENCHMARK_ONLY_TASK_GROUPS.items()},
        "tasks": [task.state_dict() for task in MAC_NATIVE_TASK_SPECS],
    }
