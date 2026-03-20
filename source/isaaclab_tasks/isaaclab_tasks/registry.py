# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static task registry specs used by the backend-aware bootstrap path."""

from __future__ import annotations

RUNTIME_REQUIREMENTS_KEY = "__runtime_requirements__"
TASK_CONTRACT_KEY = "__task_contract__"


def _isaacsim_only_kwargs(**kwargs):
    kwargs[RUNTIME_REQUIREMENTS_KEY] = {"sim_backends": ("isaacsim",)}
    return kwargs


def _with_task_contract(kwargs: dict, **contract):
    kwargs[TASK_CONTRACT_KEY] = contract
    return kwargs


MAC_SAFE_TASK_SPECS = (
    {
        "id": "Isaac-Reach-Franka-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaReachEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaReachEnvCfg",
        },
    },
    {
        "id": "Isaac-Reach-Franka-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaReachEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaReachEnvCfg",
        },
    },
    {
        "id": "Isaac-Reach-Franka-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaReachEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaReachEnvCfg",
        },
    },
    {
        "id": "Isaac-Reach-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaReachEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaReachEnvCfg",
        },
    },
    {
        "id": "Isaac-Reach-Franka-OSC-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaReachEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaReachEnvCfg",
        },
    },
    {
        "id": "Isaac-Reach-Franka-OSC-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaReachEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaReachEnvCfg",
        },
    },
    {
        "id": "Isaac-Reach-OpenArm-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmReachEnvCfg",
            },
            semantic_contract="reduced-openarm-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm reach slice preserves the single-arm reach workflow "
                "with an analytic 7-DoF surrogate instead of the exact OpenArm morphology."
            ),
        ),
    },
    {
        "id": "Isaac-Reach-OpenArm-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmReachEnvCfg",
            },
            semantic_contract="reduced-openarm-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm reach slice preserves the single-arm reach workflow "
                "with an analytic 7-DoF surrogate instead of the exact OpenArm morphology."
            ),
        ),
    },
    {
        "id": "Isaac-Reach-OpenArm-Bi-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmBiReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmBiReachEnvCfg",
            },
            semantic_contract="reduced-openarm-bimanual-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm bimanual reach slice preserves the dual-arm reach workflow "
                "with paired analytic surrogates instead of the exact OpenArm body-frame stack."
            ),
        ),
    },
    {
        "id": "Isaac-Reach-OpenArm-Bi-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmBiReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmBiReachEnvCfg",
            },
            semantic_contract="reduced-openarm-bimanual-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm bimanual reach slice preserves the dual-arm reach workflow "
                "with paired analytic surrogates instead of the exact OpenArm body-frame stack."
            ),
        ),
    },
    {
        "id": "Isaac-Reach-UR10-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10ReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10ReachEnvCfg",
            },
            semantic_contract="reduced-analytic-pose",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10 reach slice preserves the pose-tracking workflow with an "
                "analytic pose surrogate instead of the full UR10 controller stack."
            ),
        ),
    },
    {
        "id": "Isaac-Reach-UR10-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10ReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10ReachEnvCfg",
            },
            semantic_contract="reduced-analytic-pose",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10 reach slice preserves the pose-tracking workflow with an "
                "analytic pose surrogate instead of the full UR10 controller stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-Reach-UR10e-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eDeployReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eDeployReachEnvCfg",
            },
            semantic_contract="reduced-analytic-pose",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e deploy reach slice preserves the joint-space pose-command workflow "
                "with an analytic pose surrogate instead of the full deployed-robot frame transformer stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-Reach-UR10e-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eDeployReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eDeployReachEnvCfg",
            },
            semantic_contract="reduced-analytic-pose",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e deploy reach slice preserves the joint-space pose-command workflow "
                "with an analytic pose surrogate instead of the full deployed-robot frame transformer stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eDeployReachEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eDeployReachRosInferenceEnvCfg",
            },
            semantic_contract="reduced-no-ros-inference",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e deploy reach slice preserves the joint-space reach workflow, "
                "but it does not include the upstream ROS inference transport or deployed-robot runtime stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eGearAssembly2F140Env",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eGearAssembly2F140EnvCfg",
            },
            semantic_contract="reduced-analytic-assembly",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e 2F-140 gear-assembly slice preserves the shaft-alignment and insertion "
                "workflow with analytic pose tracking and scalar insertion progress instead of the full "
                "contact-rich factory assembly and ROS deployment stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eGearAssembly2F140Env",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eGearAssembly2F140EnvCfg",
            },
            semantic_contract="reduced-analytic-assembly",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e 2F-140 gear-assembly slice preserves the shaft-alignment and insertion "
                "workflow with analytic pose tracking and scalar insertion progress instead of the full "
                "contact-rich factory assembly and ROS deployment stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eGearAssembly2F85Env",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eGearAssembly2F85EnvCfg",
            },
            semantic_contract="reduced-analytic-assembly",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e 2F-85 gear-assembly slice preserves the shaft-alignment and insertion "
                "workflow with analytic pose tracking and scalar insertion progress instead of the full "
                "contact-rich factory assembly and ROS deployment stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eGearAssembly2F85Env",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10eGearAssembly2F85EnvCfg",
            },
            semantic_contract="reduced-analytic-assembly",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e 2F-85 gear-assembly slice preserves the shaft-alignment and insertion "
                "workflow with analytic pose tracking and scalar insertion progress instead of the full "
                "contact-rich factory assembly and ROS deployment stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eGearAssembly2F140Env",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": (
                    "isaaclab.backends.mac_sim.env_cfgs:MacUR10eGearAssembly2F140RosInferenceEnvCfg"
                ),
            },
            semantic_contract="reduced-no-ros-inference",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e 2F-140 gear-assembly slice preserves the shaft-alignment and insertion "
                "workflow, but it does not include the upstream ROS inference transport or deployed-robot "
                "runtime stack."
            ),
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10eGearAssembly2F85Env",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": (
                    "isaaclab.backends.mac_sim.env_cfgs:MacUR10eGearAssembly2F85RosInferenceEnvCfg"
                ),
            },
            semantic_contract="reduced-no-ros-inference",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10e 2F-85 gear-assembly slice preserves the shaft-alignment and insertion "
                "workflow, but it does not include the upstream ROS inference transport or deployed-robot "
                "runtime stack."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10LongSuctionStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10LongSuctionStackEnvCfg",
            },
            semantic_contract="reduced-analytic-suction-stack",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10 long-suction stack slice preserves the staged three-cube stacking workflow, "
                "but models the suction state as a scalar analytic surrogate instead of the upstream suction/contact stack."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacUR10ShortSuctionStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacUR10ShortSuctionStackEnvCfg",
            },
            semantic_contract="reduced-analytic-suction-stack",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native UR10 short-suction stack slice preserves the staged three-cube stacking workflow, "
                "but models the suction state as a scalar analytic surrogate instead of the upstream suction/contact stack."
            ),
        ),
    },
    {
        "id": "Isaac-Lift-Cube-Franka-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaLiftEnvCfg",
        },
    },
    {
        "id": "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaLiftEnvCfg",
        },
    },
    {
        "id": "Isaac-Lift-Cube-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaLiftEnvCfg",
        },
    },
    {
        "id": "Isaac-Lift-Cube-Franka-IK-Rel-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaLiftEnvCfg",
        },
    },
    {
        "id": "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaTeddyBearLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaTeddyBearLiftEnvCfg",
        },
    },
    {
        "id": "Isaac-Lift-Cube-Franka-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaLiftEnvCfg",
        },
    },
    {
        "id": "Isaac-Lift-Cube-OpenArm-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmLiftEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmLiftEnvCfg",
            },
            semantic_contract="reduced-openarm-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm lift slice preserves the lift workflow with a reduced "
                "analytic surrogate instead of the exact OpenArm grasp geometry."
            ),
        ),
    },
    {
        "id": "Isaac-Lift-Cube-OpenArm-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmLiftEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmLiftEnvCfg",
            },
            semantic_contract="reduced-openarm-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm lift slice preserves the lift workflow with a reduced "
                "analytic surrogate instead of the exact OpenArm grasp geometry."
            ),
        ),
    },
    {
        "id": "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacAgibotPlaceToy2BoxEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacAgibotPlaceToy2BoxEnvCfg",
            },
            semantic_contract="reduced-agibot-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native Agibot toy-to-box slice preserves the place workflow with reduced "
                "analytic grasp and placement logic instead of the exact Agibot arm and RmpFlow scene."
            ),
        ),
    },
    {
        "id": "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacAgibotPlaceUprightMugEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacAgibotPlaceUprightMugEnvCfg",
            },
            semantic_contract="reduced-agibot-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native Agibot upright-mug slice preserves the place workflow with reduced "
                "analytic grasp and placement logic instead of the exact Agibot arm, mug pose stack, and RmpFlow scene."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Franka-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
            },
            semantic_contract="reduced-analytic-stack",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native play alias resolves to the same analytic two-cube stack substrate as the "
                "other reduced Franka stack aliases; the upstream play wrapper semantics are not modeled "
                "as a separate simulator mode."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackInstanceRandomizeEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackInstanceRandomizeEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Instance-Randomize-Franka-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackInstanceRandomizeEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackInstanceRandomizeEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackRgbEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackRgbEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackRgbEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackRgbEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackVisuomotorEnvCfg",
            },
            semantic_contract="reduced-visuomotor-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native visuomotor Franka stack slice preserves the three-cube stack workflow "
                "with synthetic RGB observations and analytic object dynamics instead of the upstream "
                "robomimic image stack."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackVisuomotorCosmosEnvCfg",
            },
            semantic_contract="reduced-no-cosmos",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native visuomotor-cosmos Franka stack slice preserves the three-cube stack workflow "
                "with synthetic RGB observations, but it does not include the upstream robomimic visuomotor stack "
                "or the Cosmos multimodal image contract for RGB, segmentation, normals, and depth channels."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaBinStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaBinStackEnvCfg",
            },
            semantic_contract="reduced-no-mimic",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "Upstream mimic/imitation semantics are not implemented on mac-sim; "
                "this task resolves to the reduced bin-anchored stack slice."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Abs-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackBlueprintEnvCfg",
            },
            semantic_contract="reduced-no-blueprint",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native blueprint Franka stack slice preserves the stack workflow but does not include "
                "the upstream blueprint-conditioned generation semantics."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackSkillgenEnvCfg",
            },
            semantic_contract="reduced-no-skillgen",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native skillgen Franka stack slice preserves the stack workflow but does not include "
                "the upstream skill-generation or demonstration-conditioned behavior."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackInstanceRandomizeEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackInstanceRandomizeEnvCfg",
        },
    },
    {
        "id": "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackInstanceRandomizeEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackInstanceRandomizeEnvCfg",
        },
    },
    {
        "id": "Isaac-Franka-Cabinet-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaCabinetEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaCabinetEnvCfg",
        },
    },
    {
        "id": "Isaac-Open-Drawer-Franka-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaOpenDrawerEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaOpenDrawerEnvCfg",
        },
    },
    {
        "id": "Isaac-Open-Drawer-Franka-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaOpenDrawerEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaOpenDrawerEnvCfg",
        },
    },
    {
        "id": "Isaac-Open-Drawer-Franka-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaOpenDrawerEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaOpenDrawerEnvCfg",
        },
    },
    {
        "id": "Isaac-Open-Drawer-Franka-IK-Rel-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaOpenDrawerEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaOpenDrawerEnvCfg",
        },
    },
    {
        "id": "Isaac-Open-Drawer-OpenArm-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmOpenDrawerEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmOpenDrawerEnvCfg",
            },
            semantic_contract="reduced-openarm-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm open-drawer slice preserves the drawer-opening workflow "
                "with reduced analytic grasp logic instead of the exact cabinet scene."
            ),
        ),
    },
    {
        "id": "Isaac-Open-Drawer-OpenArm-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacOpenArmOpenDrawerEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacOpenArmOpenDrawerEnvCfg",
            },
            semantic_contract="reduced-openarm-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native OpenArm open-drawer slice preserves the drawer-opening workflow "
                "with reduced analytic grasp logic instead of the exact cabinet scene."
            ),
        ),
    },
    {
        "id": "Isaac-Franka-Cabinet-Direct-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaCabinetEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaCabinetEnvCfg",
        },
    },
    {
        "id": "Isaac-Velocity-Flat-H1-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacH1FlatEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacH1FlatEnvCfg",
        },
    },
    {
        "id": "Isaac-Velocity-Rough-H1-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacH1RoughEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacH1RoughEnvCfg",
        },
    },
    {
        "id": "Isaac-Velocity-Rough-H1-Play-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacH1RoughEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacH1RoughEnvCfg",
        },
    },
    {
        "id": "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacAnymalCFlatEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacAnymalCFlatEnvCfg",
        },
    },
    {
        "id": "Isaac-Velocity-Rough-Anymal-C-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacAnymalCRoughEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacAnymalCRoughEnvCfg",
        },
    },
    {
        "id": "Isaac-Cartpole-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacCartpoleEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacCartpoleEnvCfg",
        },
    },
    {
        "id": "Isaac-Cartpole-RGB-Camera-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacCartpoleCameraEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacCartpoleRGBCameraEnvCfg",
        },
    },
    {
        "id": "Isaac-Cartpole-Depth-Camera-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacCartpoleCameraEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacCartpoleDepthCameraEnvCfg",
        },
    },
    {
        "id": "Isaac-Cart-Double-Pendulum-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacCartDoublePendulumEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacCartDoublePendulumEnvCfg",
        },
    },
    {
        "id": "Isaac-Quadcopter-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacQuadcopterEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacQuadcopterEnvCfg",
        },
    },
    {
        "id": "Isaac-Factory-GearMesh-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim.manipulation:MacFactoryGearMeshEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFactoryGearMeshEnvCfg",
            },
            semantic_contract="reduced-analytic-gear-mesh",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native factory gear-mesh slice preserves the alignment and insertion workflow "
                "with an analytic insertion-depth surrogate instead of the upstream contact-rich factory "
                "scene and task-specific controller stack."
            ),
        ),
    },
)


ISAACSIM_ONLY_TASK_SPECS = (
    {
        "id": "Isaac-AutoMate-Assembly-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.automate.assembly_env:AssemblyEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.automate.assembly_env:AssemblyEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.automate.agents:rl_games_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-AutoMate-Disassembly-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.automate.disassembly_env:DisassemblyEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.automate.disassembly_env:DisassemblyEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.automate.agents:rl_games_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Factory-NutThread-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.factory.factory_env:FactoryEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.factory.factory_env_cfg:FactoryTaskNutThreadCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Forge-PegInsert-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.forge.forge_env:ForgeEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.forge.forge_env_cfg:ForgeTaskPegInsertCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Forge-GearMesh-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.forge.forge_env:ForgeEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.forge.forge_env_cfg:ForgeTaskGearMeshCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Forge-NutThread-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.forge.forge_env:ForgeEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.forge.forge_env_cfg:ForgeTaskNutThreadCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg_nut_thread.yaml",
        ),
    },
    {
        "id": "Isaac-Franka-Cabinet-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env:FrankaCabinetEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env:FrankaCabinetEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.franka_cabinet.agents:rl_games_ppo_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.direct.franka_cabinet.agents.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
            skrl_cfg_entry_point="isaaclab_tasks.direct.franka_cabinet.agents:skrl_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-PickPlace-GR1T2-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaBinStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaBinStackPickPlaceEnvCfg",
            },
            semantic_contract="reduced-pick-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native pick-place surrogate resolves to the reduced bin-anchored stack substrate "
                "instead of the upstream GR1T2 pick/place scene, but it keeps the reduced three-object "
                "manipulation workflow and checkpoint contract honest."
            ),
        ),
    },
    {
        "id": "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaBinStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaBinStackPickPlaceEnvCfg",
            },
            semantic_contract="reduced-pick-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native pick-place surrogate resolves to the reduced bin-anchored stack substrate "
                "instead of the upstream waist-enabled pick/place scene, but it keeps the reduced "
                "three-object manipulation workflow and checkpoint contract honest."
            ),
        ),
    },
    {
        "id": "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaBinStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaBinStackPickPlaceEnvCfg",
            },
            semantic_contract="reduced-pick-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native pick-place surrogate resolves to the reduced bin-anchored stack substrate "
                "instead of the upstream G1 Inspire FTP pick/place scene, but it keeps the reduced "
                "three-object manipulation workflow and checkpoint contract honest."
            ),
        ),
    },
    {
        "id": "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaBinStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaBinStackPickPlaceEnvCfg",
            },
            semantic_contract="reduced-pick-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native pick-place surrogate resolves to the reduced bin-anchored stack substrate "
                "instead of the upstream nut-pour scene, so the public task stays available without "
                "pretending exact cup geometry or fluid-transfer parity."
            ),
        ),
    },
    {
        "id": "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaBinStackEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaBinStackPickPlaceEnvCfg",
            },
            semantic_contract="reduced-pick-place-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native pick-place surrogate resolves to the reduced bin-anchored stack substrate "
                "instead of the upstream exhaust-pipe scene, so the public task stays available without "
                "pretending exact pipe geometry or insertion parity."
            ),
        ),
    },
    {
        "id": "Isaac-Factory-PegInsert-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim.manipulation:MacFactoryPegInsertEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFactoryPegInsertEnvCfg",
            },
            semantic_contract="reduced-analytic-peg-insert",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native factory peg-insert slice preserves the alignment and insertion workflow "
                "with an analytic insertion-depth surrogate instead of the upstream contact-rich factory "
                "scene and task-specific controller stack."
            ),
        ),
    },
    {
        "id": "Isaac-Factory-GearMesh-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim.manipulation:MacFactoryGearMeshEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFactoryGearMeshEnvCfg",
            },
            semantic_contract="reduced-analytic-gear-mesh",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native factory gear-mesh slice preserves the alignment and insertion workflow "
                "with an analytic insertion-depth surrogate instead of the upstream contact-rich factory "
                "scene and task-specific controller stack."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackVisuomotorEnvCfg",
            },
            semantic_contract="reduced-visuomotor-surrogate",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native visuomotor stack resolves to the reduced three-cube RGB stack substrate "
                "with analytic object dynamics instead of the upstream robomimic image stack."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaStackRgbEnv",
        "kwargs": _with_task_contract(
            {
                "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaStackVisuomotorCosmosEnvCfg",
            },
            semantic_contract="reduced-no-cosmos",
            upstream_alias_semantics_preserved=False,
            contract_notes=(
                "The mac-native visuomotor-cosmos stack resolves to the reduced three-cube RGB stack substrate "
                "without the upstream Cosmos multimodal image contract."
            ),
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_blueprint_env_cfg:FrankaCubeStackBlueprintEnvCfg",
        ),
    },
    {
        "id": "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_env_cfg_skillgen:FrankaCubeStackSkillgenEnvCfg",
            robomimic_bc_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.stack.config.franka.agents:robomimic/bc_rnn_low_dim.json",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Shadow-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
            skrl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:skrl_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_ff_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
            skrl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:skrl_ff_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_lstm_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env:ShadowHandVisionEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env:ShadowHandVisionEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandVisionFFPPORunnerCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_vision_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0",
        "entry_point": "isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env:ShadowHandVisionEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env:ShadowHandVisionEnvPlayCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandVisionFFPPORunnerCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_vision_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Shadow-Hand-Over-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.shadow_hand_over.shadow_hand_over_env:ShadowHandOverEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.shadow_hand_over.shadow_hand_over_env_cfg:ShadowHandOverEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.shadow_hand_over.agents:rl_games_ppo_cfg.yaml",
            skrl_cfg_entry_point="isaaclab_tasks.direct.shadow_hand_over.agents:skrl_ppo_cfg.yaml",
            skrl_ippo_cfg_entry_point="isaaclab_tasks.direct.shadow_hand_over.agents:skrl_ippo_cfg.yaml",
            skrl_mappo_cfg_entry_point="isaaclab_tasks.direct.shadow_hand_over.agents:skrl_mappo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents:rl_games_ppo_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg_PLAY",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents:rl_games_ppo_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents:rl_games_ppo_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg_PLAY",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents:rl_games_ppo_cfg.yaml",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Allegro-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg:AllegroCubeEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:rl_games_ppo_cfg.yaml",
            skrl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:skrl_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Allegro-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg:AllegroCubeEnvCfg_PLAY",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:rl_games_ppo_cfg.yaml",
            skrl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:skrl_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Allegro-NoVelObs-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg:AllegroCubeNoVelObsEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubeNoVelObsPPORunnerCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:rl_games_ppo_cfg.yaml",
            skrl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:skrl_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Repose-Cube-Allegro-NoVelObs-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg:AllegroCubeNoVelObsEnvCfg_PLAY",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg:AllegroCubeNoVelObsPPORunnerCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:rl_games_ppo_cfg.yaml",
            skrl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents:skrl_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.joint_pos_env_cfg:UR10e2F140GearAssemblyEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.agents.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.joint_pos_env_cfg:UR10e2F140GearAssemblyEnvCfg_PLAY",
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.joint_pos_env_cfg:UR10e2F85GearAssemblyEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.agents.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.joint_pos_env_cfg:UR10e2F85GearAssemblyEnvCfg_PLAY",
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.ros_inference_env_cfg:UR10e2F140GearAssemblyROSInferenceEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.agents.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.ros_inference_env_cfg:UR10e2F85GearAssemblyROSInferenceEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.ur_10e.agents.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Deploy-Reach-UR10e-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.joint_pos_env_cfg:UR10eReachEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.agents.rsl_rl_ppo_cfg:URReachPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Deploy-Reach-UR10e-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.joint_pos_env_cfg:UR10eReachEnvCfg_PLAY",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.agents.rsl_rl_ppo_cfg:URReachPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.ros_inference_env_cfg:UR10eReachROSInferenceEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.agents.rsl_rl_ppo_cfg:URReachPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Navigation-Flat-Anymal-C-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg:NavigationEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.navigation.config.anymal_c.agents.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg",
            skrl_cfg_entry_point="isaaclab_tasks.manager_based.navigation.config.anymal_c.agents:skrl_flat_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Navigation-Flat-Anymal-C-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg:NavigationEnvCfg_PLAY",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.navigation.config.anymal_c.agents.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg",
            skrl_cfg_entry_point="isaaclab_tasks.manager_based.navigation.config.anymal_c.agents:skrl_flat_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Tracking-LocoManip-Digit-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.locomanipulation.tracking.config.digit.loco_manip_env_cfg:DigitLocoManipEnvCfg",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.locomanipulation.tracking.config.digit.agents.rsl_rl_ppo_cfg:DigitLocoManipPPORunnerCfg",
        ),
    },
    {
        "id": "Isaac-Tracking-LocoManip-Digit-Play-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.locomanipulation.tracking.config.digit.loco_manip_env_cfg:DigitLocoManipEnvCfg_PLAY",
            rsl_rl_cfg_entry_point="isaaclab_tasks.manager_based.locomanipulation.tracking.config.digit.agents.rsl_rl_ppo_cfg:DigitLocoManipPPORunnerCfg",
        ),
    },
)


LAZY_IMPORT_BLACKLIST = [
    "utils",
    ".mdp",
    "pick_place",
    "direct.humanoid_amp.motions",
    "direct.automate",
    "direct.factory",
    "direct.forge",
    "direct.franka_cabinet",
    "direct.shadow_hand",
    "direct.shadow_hand_over",
    "manager_based.manipulation.dexsuite",
    "manager_based.manipulation.inhand",
    "manager_based.manipulation.deploy",
    "manager_based.navigation.config.anymal_c",
    "manager_based.locomanipulation.tracking",
]
