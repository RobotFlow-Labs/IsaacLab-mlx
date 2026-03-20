# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLX-native task wrapper surface for the mac-sim runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import mlx.core as mx

from isaaclab.backends import (
    list_public_mlx_tasks as list_supported_public_mlx_tasks,
    list_trainable_mlx_tasks as list_supported_trainable_mlx_tasks,
    mac_native_task_spec_map,
)
from isaaclab.backends.mac_sim import (
    MacAnymalCFlatEnv,
    MacAnymalCFlatEnvCfg,
    MacAnymalCRoughEnv,
    MacAnymalCRoughEnvCfg,
    MacAnymalCTrainCfg,
    MacCartpoleCameraEnv,
    MacCartpoleDepthCameraEnvCfg,
    MacCartDoublePendulumEnv,
    MacCartDoublePendulumEnvCfg,
    MacCartpoleEnvCfg,
    MacCartpoleRGBCameraEnvCfg,
    MacCartpoleTrainCfg,
    MacFrankaBinStackEnv,
    MacFrankaBinStackEnvCfg,
    MacFrankaBinStackPickPlaceEnvCfg,
    MacFrankaBinStackTrainCfg,
    MacFrankaStackBlueprintEnvCfg,
    MacFrankaCabinetEnv,
    MacFrankaCabinetEnvCfg,
    MacFrankaCabinetTrainCfg,
    MacFrankaLiftEnv,
    MacFrankaLiftEnvCfg,
    MacFrankaLiftTrainCfg,
    MacOpenArmLiftEnv,
    MacOpenArmLiftEnvCfg,
    MacOpenArmLiftTrainCfg,
    MacFrankaOpenDrawerEnv,
    MacFrankaOpenDrawerEnvCfg,
    MacFrankaOpenDrawerTrainCfg,
    MacOpenArmOpenDrawerEnv,
    MacOpenArmOpenDrawerEnvCfg,
    MacOpenArmOpenDrawerTrainCfg,
    MacOpenArmBiReachEnv,
    MacOpenArmBiReachEnvCfg,
    MacOpenArmBiReachTrainCfg,
    MacOpenArmReachEnv,
    MacOpenArmReachEnvCfg,
    MacOpenArmReachTrainCfg,
    MacFrankaReachEnv,
    MacFrankaReachEnvCfg,
    MacFrankaReachTrainCfg,
    MacUR10ReachEnv,
    MacUR10ReachEnvCfg,
    MacUR10ReachTrainCfg,
    MacUR10LongSuctionStackEnv,
    MacUR10LongSuctionStackEnvCfg,
    MacUR10LongSuctionStackTrainCfg,
    MacUR10ShortSuctionStackEnv,
    MacUR10ShortSuctionStackEnvCfg,
    MacUR10ShortSuctionStackTrainCfg,
    MacUR10eGearAssembly2F140Env,
    MacUR10eGearAssembly2F140EnvCfg,
    MacUR10eGearAssembly2F140RosInferenceEnvCfg,
    MacUR10eGearAssembly2F140TrainCfg,
    MacUR10eGearAssembly2F85Env,
    MacUR10eGearAssembly2F85EnvCfg,
    MacUR10eGearAssembly2F85RosInferenceEnvCfg,
    MacUR10eGearAssembly2F85TrainCfg,
    MacUR10eDeployReachEnv,
    MacUR10eDeployReachEnvCfg,
    MacUR10eDeployReachRosInferenceEnvCfg,
    MacUR10eDeployReachTrainCfg,
    MacFrankaStackEnv,
    MacFrankaStackEnvCfg,
    MacFrankaStackInstanceRandomizeEnv,
    MacFrankaStackInstanceRandomizeEnvCfg,
    MacFrankaStackInstanceRandomizeTrainCfg,
    MacFrankaStackRgbEnv,
    MacFrankaStackRgbEnvCfg,
    MacFrankaStackRgbTrainCfg,
    MacFrankaStackSkillgenEnvCfg,
    MacFrankaStackTrainCfg,
    MacFrankaStackVisuomotorCosmosEnvCfg,
    MacFrankaStackVisuomotorEnvCfg,
    MacFrankaTeddyBearLiftEnv,
    MacFrankaTeddyBearLiftEnvCfg,
    MacFrankaTeddyBearLiftTrainCfg,
    MacH1FlatEnv,
    MacH1FlatEnvCfg,
    MacH1RoughEnv,
    MacH1RoughEnvCfg,
    MacH1TrainCfg,
    MacQuadcopterEnv,
    MacQuadcopterEnvCfg,
    play_anymal_c_policy,
    play_cartpole_policy,
    play_franka_bin_stack_policy,
    play_franka_cabinet_policy,
    play_franka_lift_policy,
    play_openarm_lift_policy,
    play_franka_open_drawer_policy,
    play_openarm_open_drawer_policy,
    play_openarm_bi_reach_policy,
    play_openarm_reach_policy,
    play_franka_reach_policy,
    play_franka_stack_blueprint_policy,
    play_franka_stack_skillgen_policy,
    play_franka_stack_visuomotor_cosmos_policy,
    play_franka_stack_visuomotor_policy,
    play_ur10_long_suction_stack_policy,
    play_ur10_reach_policy,
    play_ur10_short_suction_stack_policy,
    play_ur10e_gear_assembly_2f140_policy,
    play_ur10e_gear_assembly_2f85_policy,
    play_ur10e_deploy_reach_policy,
    play_ur10e_deploy_reach_ros_inference_policy,
    play_franka_stack_policy,
    play_franka_stack_instance_randomize_policy,
    play_franka_stack_rgb_policy,
    play_franka_teddy_bear_lift_policy,
    play_h1_policy,
    resolve_resume_hidden_dim,
    train_anymal_c_policy,
    train_cartpole_policy,
    train_franka_bin_stack_policy,
    train_franka_cabinet_policy,
    train_franka_lift_policy,
    train_openarm_lift_policy,
    train_franka_open_drawer_policy,
    train_openarm_open_drawer_policy,
    train_openarm_bi_reach_policy,
    train_openarm_reach_policy,
    train_franka_reach_policy,
    train_franka_stack_blueprint_policy,
    train_franka_stack_skillgen_policy,
    train_franka_stack_visuomotor_cosmos_policy,
    train_franka_stack_visuomotor_policy,
    train_ur10_long_suction_stack_policy,
    train_ur10_reach_policy,
    train_ur10_short_suction_stack_policy,
    train_ur10e_gear_assembly_2f140_policy,
    train_ur10e_gear_assembly_2f85_policy,
    train_ur10e_deploy_reach_policy,
    train_ur10e_deploy_reach_ros_inference_policy,
    train_franka_stack_policy,
    train_franka_stack_instance_randomize_policy,
    train_franka_stack_rgb_policy,
    train_franka_teddy_bear_lift_policy,
    train_h1_policy,
)

TRAINABLE_MLX_TASKS = list_supported_trainable_mlx_tasks()
PUBLIC_MLX_TASKS = list_supported_public_mlx_tasks()
PICK_PLACE_ALIAS_TASKS = {
    "Isaac-PickPlace-GR1T2-Abs-v0",
    "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
    "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
    "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
    "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
}


@dataclass(frozen=True)
class MlxTaskSpec:
    """Stable public task metadata for MLX/mac-sim runners."""

    task: str
    trainable: bool
    default_checkpoint: str | None
    default_hidden_dim: int | None
    default_action_std: float | None = None
    semantic_contract: str = "aligned"
    upstream_alias_semantics_preserved: bool = True
    notes: str = ""


_SUPPORTED_TASK_SPECS = mac_native_task_spec_map()
MLX_TASK_SPECS = {
    task_id: MlxTaskSpec(
        task=task_id,
        trainable=spec.trainable,
        default_checkpoint=spec.default_checkpoint,
        default_hidden_dim=spec.default_hidden_dim,
        default_action_std=spec.default_action_std,
        semantic_contract=spec.semantic_contract,
        upstream_alias_semantics_preserved=spec.upstream_alias_semantics_preserved,
        notes=spec.notes,
    )
    for task_id, spec in _SUPPORTED_TASK_SPECS.items()
}

MLX_TASK_ALIASES: dict[str, str] = {
    "Isaac-Reach-Franka-v0": "franka-reach",
    "Isaac-Reach-Franka-Play-v0": "franka-reach",
    "Isaac-Reach-Franka-IK-Abs-v0": "franka-reach",
    "Isaac-Reach-Franka-IK-Rel-v0": "franka-reach",
    "Isaac-Reach-Franka-OSC-v0": "franka-reach",
    "Isaac-Reach-Franka-OSC-Play-v0": "franka-reach",
    "Isaac-Reach-OpenArm-v0": "openarm-reach",
    "Isaac-Reach-OpenArm-Play-v0": "openarm-reach",
    "Isaac-Reach-OpenArm-Bi-v0": "openarm-bi-reach",
    "Isaac-Reach-OpenArm-Bi-Play-v0": "openarm-bi-reach",
    "Isaac-Reach-UR10-v0": "ur10-reach",
    "Isaac-Reach-UR10-Play-v0": "ur10-reach",
    "Isaac-Deploy-GearAssembly-UR10e-2F140-v0": "ur10e-gear-assembly-2f140",
    "Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0": "ur10e-gear-assembly-2f140",
    "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0": "ur10e-gear-assembly-2f140",
    "Isaac-Deploy-GearAssembly-UR10e-2F85-v0": "ur10e-gear-assembly-2f85",
    "Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0": "ur10e-gear-assembly-2f85",
    "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0": "ur10e-gear-assembly-2f85",
    "Isaac-Deploy-Reach-UR10e-v0": "ur10e-deploy-reach",
    "Isaac-Deploy-Reach-UR10e-Play-v0": "ur10e-deploy-reach",
    "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0": "ur10e-deploy-reach",
    "Isaac-Lift-Cube-Franka-v0": "franka-lift",
    "Isaac-Lift-Cube-Franka-Play-v0": "franka-lift",
    "Isaac-Lift-Cube-Franka-IK-Abs-v0": "franka-lift",
    "Isaac-Lift-Cube-Franka-IK-Rel-v0": "franka-lift",
    "Isaac-Lift-Cube-OpenArm-v0": "openarm-lift",
    "Isaac-Lift-Cube-OpenArm-Play-v0": "openarm-lift",
    "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0": "franka-teddy-bear-lift",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-v0": "franka-stack-instance-randomize",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0": "franka-stack-instance-randomize",
    "Isaac-Stack-Cube-Franka-v0": "franka-stack",
    "Isaac-Stack-Cube-Franka-Play-v0": "franka-stack",
    "Isaac-Stack-Cube-Franka-IK-Rel-v0": "franka-stack",
    "Isaac-Stack-Cube-Franka-IK-Abs-v0": "franka-stack",
    "Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0": "franka-stack",
    "Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0": "franka-stack",
    "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0": "franka-stack-rgb",
    "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-Play-v0": "franka-stack-rgb",
    "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0": "franka-stack-rgb",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0": "franka-stack-rgb",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0": "franka-stack-rgb",
    "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0": "franka-stack",
    "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0": "franka-stack",
    "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0": "franka-bin-stack",
    "Isaac-PickPlace-GR1T2-Abs-v0": "franka-bin-stack",
    "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0": "franka-bin-stack",
    "Isaac-PickPlace-G1-InspireFTP-Abs-v0": "franka-bin-stack",
    "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0": "franka-bin-stack",
    "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0": "franka-bin-stack",
    "Isaac-Franka-Cabinet-Direct-v0": "franka-cabinet",
    "Isaac-Franka-Cabinet-Direct-Play-v0": "franka-cabinet",
    "Isaac-Open-Drawer-Franka-v0": "franka-open-drawer",
    "Isaac-Open-Drawer-Franka-Play-v0": "franka-open-drawer",
    "Isaac-Open-Drawer-Franka-IK-Abs-v0": "franka-open-drawer",
    "Isaac-Open-Drawer-Franka-IK-Rel-v0": "franka-open-drawer",
    "Isaac-Open-Drawer-OpenArm-v0": "openarm-open-drawer",
    "Isaac-Open-Drawer-OpenArm-Play-v0": "openarm-open-drawer",
}

MLX_ALIAS_TASK_SPECS: dict[str, MlxTaskSpec] = {
    "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0": replace(
        MLX_TASK_SPECS["ur10e-deploy-reach"],
        task="Isaac-Deploy-Reach-UR10e-ROS-Inference-v0",
        semantic_contract="reduced-no-ros-inference",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10e deploy-reach slice without the upstream ROS inference transport "
            "or deployed-robot runtime stack."
        ),
    ),
    "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0": replace(
        MLX_TASK_SPECS["ur10e-gear-assembly-2f140"],
        task="Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0",
        semantic_contract="reduced-no-ros-inference",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10e 2F-140 gear-assembly slice without the upstream ROS inference "
            "transport or deployed-robot runtime stack."
        ),
    ),
    "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0": replace(
        MLX_TASK_SPECS["ur10e-gear-assembly-2f85"],
        task="Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0",
        semantic_contract="reduced-no-ros-inference",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic UR10e 2F-85 gear-assembly slice without the upstream ROS inference "
            "transport or deployed-robot runtime stack."
        ),
    ),
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0": replace(
        MLX_TASK_SPECS["franka-stack-rgb"],
        task="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        semantic_contract="reduced-visuomotor-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced three-cube Franka stack slice with synthetic RGB observations and analytic object "
            "dynamics instead of the upstream robomimic visuomotor stack."
        ),
    ),
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0": replace(
        MLX_TASK_SPECS["franka-stack-rgb"],
        task="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
        semantic_contract="reduced-no-cosmos",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced three-cube Franka stack slice with synthetic RGB observations, without the "
            "upstream robomimic visuomotor stack or the Cosmos multimodal image contract."
        ),
    ),
    "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0": replace(
        MLX_TASK_SPECS["franka-stack"],
        task="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
        semantic_contract="reduced-no-blueprint",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic two-cube Franka stack slice without the upstream blueprint-conditioned generation semantics."
        ),
    ),
    "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0": replace(
        MLX_TASK_SPECS["franka-stack"],
        task="Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
        semantic_contract="reduced-no-skillgen",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced analytic two-cube Franka stack slice without the upstream skill-generation or "
            "demonstration-conditioned behavior."
        ),
    ),
    "Isaac-PickPlace-GR1T2-Abs-v0": replace(
        MLX_TASK_SPECS["franka-bin-stack"],
        task="Isaac-PickPlace-GR1T2-Abs-v0",
        semantic_contract="reduced-pick-place-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced pick-place surrogate mapped onto the bin-anchored stack substrate instead of the "
            "upstream GR1T2 pick/place scene."
        ),
    ),
    "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0": replace(
        MLX_TASK_SPECS["franka-bin-stack"],
        task="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
        semantic_contract="reduced-pick-place-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced pick-place surrogate mapped onto the bin-anchored stack substrate instead of the "
            "upstream waist-enabled pick/place scene."
        ),
    ),
    "Isaac-PickPlace-G1-InspireFTP-Abs-v0": replace(
        MLX_TASK_SPECS["franka-bin-stack"],
        task="Isaac-PickPlace-G1-InspireFTP-Abs-v0",
        semantic_contract="reduced-pick-place-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced pick-place surrogate mapped onto the bin-anchored stack substrate instead of the "
            "upstream G1 Inspire FTP pick/place scene."
        ),
    ),
    "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0": replace(
        MLX_TASK_SPECS["franka-bin-stack"],
        task="Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
        semantic_contract="reduced-pick-place-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced pick-place surrogate mapped onto the bin-anchored stack substrate instead of the "
            "upstream nut-pour scene."
        ),
    ),
    "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0": replace(
        MLX_TASK_SPECS["franka-bin-stack"],
        task="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
        semantic_contract="reduced-pick-place-surrogate",
        upstream_alias_semantics_preserved=False,
        notes=(
            "Reduced pick-place surrogate mapped onto the bin-anchored stack substrate instead of the "
            "upstream exhaust-pipe scene."
        ),
    ),
}


def _serialize_task_spec(spec: MlxTaskSpec) -> dict[str, Any]:
    """Return a JSON-safe task spec payload for CLI and artifact surfaces."""

    return asdict(spec)


def _with_task_spec(payload: dict[str, Any], spec: MlxTaskSpec) -> dict[str, Any]:
    """Attach the normalized public task spec to an eval/train payload."""

    if "task_spec" not in payload:
        payload["task_spec"] = _serialize_task_spec(spec)
    return payload


def _normalize_mlx_task(task: str) -> str:
    """Normalize upstream-compatible task aliases to canonical MLX/mac task ids."""

    return MLX_TASK_ALIASES.get(task, task)


def list_mlx_tasks() -> tuple[str, ...]:
    """Return the currently supported MLX/mac-sim task ids."""

    return PUBLIC_MLX_TASKS


def list_trainable_mlx_tasks() -> tuple[str, ...]:
    """Return the MLX/mac-sim task ids with a train surface."""

    return TRAINABLE_MLX_TASKS


def get_mlx_task_spec(task: str) -> MlxTaskSpec:
    """Return the stable MLX task metadata for a supported task."""

    if task in MLX_ALIAS_TASK_SPECS:
        return MLX_ALIAS_TASK_SPECS[task]
    task = _normalize_mlx_task(task)
    try:
        return MLX_TASK_SPECS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported MLX task: {task}") from exc


def train_mlx_task(
    task: str,
    *,
    num_envs: int = 256,
    updates: int = 10,
    rollout_steps: int = 24,
    epochs_per_update: int = 2,
    learning_rate: float = 3e-4,
    hidden_dim: int | None = None,
    action_std: float | None = None,
    checkpoint: str | None = None,
    resume_from: str | None = None,
    eval_interval: int = 5,
    episode_length_s: float = 20.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a supported MLX/mac-sim task and return a normalized payload."""

    requested_task = task
    task = _normalize_mlx_task(task)
    spec = get_mlx_task_spec(requested_task)
    if not spec.trainable:
        raise ValueError(f"Task '{task}' does not expose an MLX training surface.")

    if task == "cartpole":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacCartpoleTrainCfg(
            env=MacCartpoleEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/cartpole_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_cartpole_policy(cfg)
    elif task == "anymal-c-flat":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacAnymalCTrainCfg(
            env=MacAnymalCFlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/anymal_c_flat_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_anymal_c_policy(cfg)
    elif task == "anymal-c-rough":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacAnymalCTrainCfg(
            env=MacAnymalCRoughEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/anymal_c_rough_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_anymal_c_policy(cfg)
    elif task == "h1-flat":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 192)
        )
        cfg = MacH1TrainCfg(
            env=MacH1FlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/h1_flat_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_h1_policy(cfg)
    elif task == "h1-rough":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 192)
        )
        cfg = MacH1TrainCfg(
            env=MacH1RoughEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/h1_rough_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_h1_policy(cfg)
    elif task == "franka-reach":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacFrankaReachTrainCfg(
            env=MacFrankaReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_reach_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_reach_policy(cfg)
    elif task == "openarm-reach":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacOpenArmReachTrainCfg(
            env=MacOpenArmReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/openarm_reach_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_openarm_reach_policy(cfg)
    elif task == "openarm-bi-reach":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 160)
        )
        cfg = MacOpenArmBiReachTrainCfg(
            env=MacOpenArmBiReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/openarm_bi_reach_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_openarm_bi_reach_policy(cfg)
    elif task == "ur10-reach":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacUR10ReachTrainCfg(
            env=MacUR10ReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/ur10_reach_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_ur10_reach_policy(cfg)
    elif task == "ur10-long-suction-stack":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacUR10LongSuctionStackTrainCfg(
            env=MacUR10LongSuctionStackEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/ur10_long_suction_stack_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_ur10_long_suction_stack_policy(cfg)
    elif task == "ur10-short-suction-stack":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacUR10ShortSuctionStackTrainCfg(
            env=MacUR10ShortSuctionStackEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/ur10_short_suction_stack_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_ur10_short_suction_stack_policy(cfg)
    elif task == "ur10e-gear-assembly-2f140":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        env_cfg = (
            MacUR10eGearAssembly2F140RosInferenceEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0"
            else MacUR10eGearAssembly2F140EnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        cfg = MacUR10eGearAssembly2F140TrainCfg(
            env=env_cfg,
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/ur10e_gear_assembly_2f140_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_ur10e_gear_assembly_2f140_policy(cfg)
    elif task == "ur10e-gear-assembly-2f85":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        env_cfg = (
            MacUR10eGearAssembly2F85RosInferenceEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0"
            else MacUR10eGearAssembly2F85EnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        cfg = MacUR10eGearAssembly2F85TrainCfg(
            env=env_cfg,
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/ur10e_gear_assembly_2f85_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_ur10e_gear_assembly_2f85_policy(cfg)
    elif task == "ur10e-deploy-reach":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        env_cfg = (
            MacUR10eDeployReachRosInferenceEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0"
            else MacUR10eDeployReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        cfg = MacUR10eDeployReachTrainCfg(
            env=env_cfg,
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/ur10e_deploy_reach_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = (
            train_ur10e_deploy_reach_ros_inference_policy(cfg)
            if requested_task == "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0"
            else train_ur10e_deploy_reach_policy(cfg)
        )
    elif task == "franka-lift":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacFrankaLiftTrainCfg(
            env=MacFrankaLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_lift_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_lift_policy(cfg)
    elif task == "openarm-lift":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacOpenArmLiftTrainCfg(
            env=MacOpenArmLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/openarm_lift_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_openarm_lift_policy(cfg)
    elif task == "franka-teddy-bear-lift":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacFrankaTeddyBearLiftTrainCfg(
            env=MacFrankaTeddyBearLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_teddy_bear_lift_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_teddy_bear_lift_policy(cfg)
    elif task == "franka-stack-instance-randomize":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacFrankaStackInstanceRandomizeTrainCfg(
            env=MacFrankaStackInstanceRandomizeEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint
            or spec.default_checkpoint
            or "logs/mlx/franka_stack_instance_randomize_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_stack_instance_randomize_policy(cfg)
    elif task == "franka-stack":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0":
            env_cfg = MacFrankaStackBlueprintEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        elif requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0":
            env_cfg = MacFrankaStackSkillgenEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        else:
            env_cfg = MacFrankaStackEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        cfg = MacFrankaStackTrainCfg(
            env=env_cfg,
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_stack_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0":
            result = train_franka_stack_blueprint_policy(cfg)
        elif requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0":
            result = train_franka_stack_skillgen_policy(cfg)
        else:
            result = train_franka_stack_policy(cfg)
    elif task == "franka-stack-rgb":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0":
            env_cfg = MacFrankaStackVisuomotorEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        elif requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0":
            env_cfg = MacFrankaStackVisuomotorCosmosEnvCfg(
                num_envs=num_envs,
                seed=seed,
                episode_length_s=episode_length_s,
            )
        else:
            env_cfg = MacFrankaStackRgbEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        cfg = MacFrankaStackRgbTrainCfg(
            env=env_cfg,
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_stack_rgb_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0":
            result = train_franka_stack_visuomotor_policy(cfg)
        elif requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0":
            result = train_franka_stack_visuomotor_cosmos_policy(cfg)
        else:
            result = train_franka_stack_rgb_policy(cfg)
    elif task == "franka-bin-stack":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        bin_stack_env_cfg_cls = (
            MacFrankaBinStackPickPlaceEnvCfg if requested_task in PICK_PLACE_ALIAS_TASKS else MacFrankaBinStackEnvCfg
        )
        cfg = MacFrankaBinStackTrainCfg(
            env=bin_stack_env_cfg_cls(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_bin_stack_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_bin_stack_policy(cfg)
    elif task == "franka-cabinet":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacFrankaCabinetTrainCfg(
            env=MacFrankaCabinetEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_cabinet_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_cabinet_policy(cfg)
    elif task == "franka-open-drawer":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacFrankaOpenDrawerTrainCfg(
            env=MacFrankaOpenDrawerEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/franka_open_drawer_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_franka_open_drawer_policy(cfg)
    elif task == "openarm-open-drawer":
        resolved_hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else resolve_resume_hidden_dim(resume_from, spec.default_hidden_dim or 128)
        )
        cfg = MacOpenArmOpenDrawerTrainCfg(
            env=MacOpenArmOpenDrawerEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s),
            hidden_dim=resolved_hidden_dim,
            updates=updates,
            rollout_steps=rollout_steps,
            epochs_per_update=epochs_per_update,
            learning_rate=learning_rate,
            action_std=spec.default_action_std if action_std is None else action_std,
            checkpoint_path=checkpoint or spec.default_checkpoint or "logs/mlx/openarm_open_drawer_policy.npz",
            resume_from=resume_from,
            eval_interval=eval_interval,
        )
        result = train_openarm_open_drawer_policy(cfg)
    else:
        raise ValueError(f"Unsupported MLX training task: {task}")

    return {
        "task": task,
        "task_spec": _serialize_task_spec(spec),
        "train_cfg": result["train_cfg"],
        "checkpoint_path": result["checkpoint_path"],
        "metadata_path": result["metadata_path"],
        "resumed_from": result["resumed_from"],
        "completed_episodes": result["completed_episodes"],
        "mean_recent_return": float(result["mean_recent_return"]),
    }


def evaluate_mlx_task(
    task: str,
    *,
    num_envs: int = 64,
    episodes: int = 3,
    seed: int = 42,
    episode_length_s: float = 20.0,
    max_steps: int = 10000,
    checkpoint: str | None = None,
    hidden_dim: int | None = None,
    random_actions: bool = False,
    cart_action: float = 0.0,
    pendulum_action: float = 0.0,
    thrust_action: float = 0.2,
    roll_action: float = 0.0,
    pitch_action: float = 0.0,
    yaw_action: float = 0.0,
) -> dict[str, Any]:
    """Evaluate or replay a supported MLX/mac-sim task."""

    requested_task = task
    task = _normalize_mlx_task(task)
    try:
        spec = get_mlx_task_spec(requested_task)
    except ValueError as exc:
        raise ValueError(f"Unsupported MLX evaluation task: {requested_task}") from exc
    if task == "cartpole":
        if checkpoint is None:
            raise ValueError("Cartpole evaluation requires a checkpoint.")
        returns = play_cartpole_policy(
            checkpoint,
            env_cfg=MacCartpoleEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
            episodes=episodes,
            hidden_dim=hidden_dim,
        )
        return {
            "task": task,
            "mode": "checkpoint",
            "episodes_requested": episodes,
            "episodes_completed": len(returns),
            "completed": [{"return": float(value)} for value in returns],
            "checkpoint": checkpoint,
        }

    if task == "cartpole-rgb-camera":
        if checkpoint is not None:
            raise ValueError("Cartpole RGB camera does not expose checkpoint replay on the public MLX wrapper.")
        env = MacCartpoleCameraEnv(
            MacCartpoleRGBCameraEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1))
                if random_actions
                else mx.full((num_envs, 1), cart_action, dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "cartpole-depth-camera":
        if checkpoint is not None:
            raise ValueError("Cartpole depth camera does not expose checkpoint replay on the public MLX wrapper.")
        env = MacCartpoleCameraEnv(
            MacCartpoleDepthCameraEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1))
                if random_actions
                else mx.full((num_envs, 1), cart_action, dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "anymal-c-flat":
        if checkpoint is not None:
            returns = play_anymal_c_policy(
                checkpoint,
                env_cfg=MacAnymalCFlatEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacAnymalCFlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacAnymalCFlatEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "h1-flat":
        if checkpoint is not None:
            returns = play_h1_policy(
                checkpoint,
                env_cfg=MacH1FlatEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacH1FlatEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacH1FlatEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "h1-rough":
        if checkpoint is not None:
            returns = play_h1_policy(
                checkpoint,
                env_cfg=MacH1RoughEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacH1RoughEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacH1RoughEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "anymal-c-rough":
        if checkpoint is not None:
            returns = play_anymal_c_policy(
                checkpoint,
                env_cfg=MacAnymalCRoughEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacAnymalCRoughEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacAnymalCRoughEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "franka-reach":
        if checkpoint is not None:
            returns = play_franka_reach_policy(
                checkpoint,
                env_cfg=MacFrankaReachEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacFrankaReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaReachEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "openarm-reach":
        if checkpoint is not None:
            returns = play_openarm_reach_policy(
                checkpoint,
                env_cfg=MacOpenArmReachEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacOpenArmReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacOpenArmReachEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "openarm-bi-reach":
        if checkpoint is not None:
            returns = play_openarm_bi_reach_policy(
                checkpoint,
                env_cfg=MacOpenArmBiReachEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacOpenArmBiReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacOpenArmBiReachEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "ur10-reach":
        if checkpoint is not None:
            returns = play_ur10_reach_policy(
                checkpoint,
                env_cfg=MacUR10ReachEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacUR10ReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacUR10ReachEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "ur10-long-suction-stack":
        if checkpoint is not None:
            returns = play_ur10_long_suction_stack_policy(
                checkpoint,
                env_cfg=MacUR10LongSuctionStackEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacUR10LongSuctionStackEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacUR10LongSuctionStackEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "ur10-short-suction-stack":
        if checkpoint is not None:
            returns = play_ur10_short_suction_stack_policy(
                checkpoint,
                env_cfg=MacUR10ShortSuctionStackEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacUR10ShortSuctionStackEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacUR10ShortSuctionStackEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "ur10e-gear-assembly-2f140":
        if checkpoint is not None:
            returns = play_ur10e_gear_assembly_2f140_policy(
                checkpoint,
                env_cfg=(
                    MacUR10eGearAssembly2F140RosInferenceEnvCfg(
                        num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                    )
                    if requested_task == "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0"
                    else MacUR10eGearAssembly2F140EnvCfg(
                        num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                    )
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = (
            MacUR10eGearAssembly2F140RosInferenceEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0"
            else MacUR10eGearAssembly2F140EnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        env = MacUR10eGearAssembly2F140Env(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "ur10e-gear-assembly-2f85":
        if checkpoint is not None:
            returns = play_ur10e_gear_assembly_2f85_policy(
                checkpoint,
                env_cfg=(
                    MacUR10eGearAssembly2F85RosInferenceEnvCfg(
                        num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                    )
                    if requested_task == "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0"
                    else MacUR10eGearAssembly2F85EnvCfg(
                        num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                    )
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = (
            MacUR10eGearAssembly2F85RosInferenceEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0"
            else MacUR10eGearAssembly2F85EnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        env = MacUR10eGearAssembly2F85Env(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "ur10e-deploy-reach":
        if checkpoint is not None:
            env_cfg = (
                MacUR10eDeployReachRosInferenceEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                )
                if requested_task == "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0"
                else MacUR10eDeployReachEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s)
            )
            returns = (
                play_ur10e_deploy_reach_ros_inference_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
                if requested_task == "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0"
                else play_ur10e_deploy_reach_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = (
            MacUR10eDeployReachRosInferenceEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0"
            else MacUR10eDeployReachEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        env = MacUR10eDeployReachEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "franka-lift":
        if checkpoint is not None:
            returns = play_franka_lift_policy(
                checkpoint,
                env_cfg=MacFrankaLiftEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = MacFrankaLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaLiftEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "openarm-lift":
        if checkpoint is not None:
            returns = play_openarm_lift_policy(
                checkpoint,
                env_cfg=MacOpenArmLiftEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacOpenArmLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacOpenArmLiftEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-teddy-bear-lift":
        if checkpoint is not None:
            returns = play_franka_teddy_bear_lift_policy(
                checkpoint,
                env_cfg=MacFrankaTeddyBearLiftEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacFrankaTeddyBearLiftEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaTeddyBearLiftEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-stack-instance-randomize":
        if checkpoint is not None:
            returns = play_franka_stack_instance_randomize_policy(
                checkpoint,
                env_cfg=MacFrankaStackInstanceRandomizeEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacFrankaStackInstanceRandomizeEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaStackInstanceRandomizeEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-stack":
        if checkpoint is not None:
            env_cfg = (
                MacFrankaStackBlueprintEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s)
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0"
                else MacFrankaStackSkillgenEnvCfg(
                    num_envs=max(1, num_envs),
                    seed=seed,
                    episode_length_s=episode_length_s,
                )
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"
                else MacFrankaStackEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s)
            )
            returns = (
                play_franka_stack_blueprint_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0"
                else play_franka_stack_skillgen_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"
                else play_franka_stack_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = (
            MacFrankaStackBlueprintEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0"
            else MacFrankaStackSkillgenEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0"
            else MacFrankaStackEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        env = MacFrankaStackEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "franka-stack-rgb":
        if checkpoint is not None:
            env_cfg = (
                MacFrankaStackVisuomotorEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s)
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
                else MacFrankaStackVisuomotorCosmosEnvCfg(
                    num_envs=max(1, num_envs),
                    seed=seed,
                    episode_length_s=episode_length_s,
                )
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0"
                else MacFrankaStackRgbEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s)
            )
            returns = (
                play_franka_stack_visuomotor_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
                else play_franka_stack_visuomotor_cosmos_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
                if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0"
                else play_franka_stack_rgb_policy(
                    checkpoint,
                    env_cfg=env_cfg,
                    episodes=episodes,
                    hidden_dim=hidden_dim,
                )
            )
            return _with_task_spec(
                {
                    "task": task,
                    "mode": "checkpoint",
                    "episodes_requested": episodes,
                    "episodes_completed": len(returns),
                    "completed": [{"return": float(value)} for value in returns],
                    "checkpoint": checkpoint,
                },
                spec,
            )
        cfg = (
            MacFrankaStackVisuomotorEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
            else MacFrankaStackVisuomotorCosmosEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
            if requested_task == "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0"
            else MacFrankaStackRgbEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        env = MacFrankaStackRgbEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return _with_task_spec(
            {
                "task": task,
                "mode": "manual",
                "episodes_requested": episodes,
                "episodes_completed": len(completed[:episodes]),
                "completed": completed[:episodes],
                "max_steps": max_steps,
            },
            spec,
        )

    if task == "franka-bin-stack":
        if checkpoint is not None:
            bin_stack_env_cfg_cls = (
                MacFrankaBinStackPickPlaceEnvCfg if requested_task in PICK_PLACE_ALIAS_TASKS else MacFrankaBinStackEnvCfg
            )
            returns = play_franka_bin_stack_policy(
                checkpoint,
                env_cfg=bin_stack_env_cfg_cls(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        bin_stack_env_cfg_cls = (
            MacFrankaBinStackPickPlaceEnvCfg if requested_task in PICK_PLACE_ALIAS_TASKS else MacFrankaBinStackEnvCfg
        )
        cfg = bin_stack_env_cfg_cls(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaBinStackEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-cabinet":
        if checkpoint is not None:
            returns = play_franka_cabinet_policy(
                checkpoint,
                env_cfg=MacFrankaCabinetEnvCfg(num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacFrankaCabinetEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaCabinetEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "franka-open-drawer":
        if checkpoint is not None:
            returns = play_franka_open_drawer_policy(
                checkpoint,
                env_cfg=MacFrankaOpenDrawerEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacFrankaOpenDrawerEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacFrankaOpenDrawerEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "cart-double-pendulum":
        if checkpoint is not None:
            raise ValueError("Cart-double-pendulum does not expose checkpoint replay on the public MLX wrapper.")
        env = MacCartDoublePendulumEnv(
            MacCartDoublePendulumEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        )
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = {
                "cart": (
                    mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1))
                    if random_actions
                    else mx.full((num_envs, 1), cart_action, dtype=mx.float32)
                ),
                "pendulum": (
                    mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 1))
                    if random_actions
                    else mx.full((num_envs, 1), pendulum_action, dtype=mx.float32)
                ),
            }
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {
                    "cart_return": float(cart_return),
                    "pendulum_return": float(pendulum_return),
                    "length": int(length),
                }
                for cart_return, pendulum_return, length in zip(
                    extras.get("completed_returns", {}).get("cart", []),
                    extras.get("completed_returns", {}).get("pendulum", []),
                    extras.get("completed_lengths", []),
                    strict=True,
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "quadcopter":
        if checkpoint is not None:
            raise ValueError("Quadcopter does not expose checkpoint replay on the public MLX wrapper.")
        env = MacQuadcopterEnv(MacQuadcopterEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s))
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(num_envs, 4))
                if random_actions
                else mx.array([[thrust_action, roll_action, pitch_action, yaw_action]] * num_envs, dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "final_distance": float(extras["final_distance_to_goal"])}
                for length in extras.get("completed_lengths", [])
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    if task == "openarm-open-drawer":
        if checkpoint is not None:
            returns = play_openarm_open_drawer_policy(
                checkpoint,
                env_cfg=MacOpenArmOpenDrawerEnvCfg(
                    num_envs=max(1, num_envs), seed=seed, episode_length_s=episode_length_s
                ),
                episodes=episodes,
                hidden_dim=hidden_dim,
            )
            return {
                "task": task,
                "mode": "checkpoint",
                "episodes_requested": episodes,
                "episodes_completed": len(returns),
                "completed": [{"return": float(value)} for value in returns],
                "checkpoint": checkpoint,
            }
        cfg = MacOpenArmOpenDrawerEnvCfg(num_envs=num_envs, seed=seed, episode_length_s=episode_length_s)
        env = MacOpenArmOpenDrawerEnv(cfg)
        mx.random.seed(seed)
        env.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(max_steps):
            actions = (
                mx.random.uniform(low=-1.0, high=1.0, shape=(cfg.num_envs, cfg.action_space))
                if random_actions
                else mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)
            )
            _, _, _, _, extras = env.step(actions)
            completed.extend(
                {"length": int(length), "return": float(value)}
                for length, value in zip(
                    extras.get("completed_lengths", []), extras.get("completed_returns", []), strict=True
                )
            )
            if len(completed) >= episodes:
                break
        return {
            "task": task,
            "mode": "manual",
            "episodes_requested": episodes,
            "episodes_completed": len(completed[:episodes]),
            "completed": completed[:episodes],
            "max_steps": max_steps,
        }

    raise ValueError(f"Unsupported MLX evaluation task: {task}")
