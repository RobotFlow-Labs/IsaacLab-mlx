# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static task registry specs used by the backend-aware bootstrap path."""

from __future__ import annotations

RUNTIME_REQUIREMENTS_KEY = "__runtime_requirements__"


def _isaacsim_only_kwargs(**kwargs):
    kwargs[RUNTIME_REQUIREMENTS_KEY] = {"sim_backends": ("isaacsim",)}
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
        "id": "Isaac-Lift-Cube-Franka-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacFrankaLiftEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim.env_cfgs:MacFrankaLiftEnvCfg",
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
        "id": "Isaac-Factory-PegInsert-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.factory.factory_env:FactoryEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.factory.factory_env_cfg:FactoryTaskPegInsertCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
        ),
    },
    {
        "id": "Isaac-Factory-GearMesh-Direct-v0",
        "entry_point": "isaaclab_tasks.direct.factory.factory_env:FactoryEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.direct.factory.factory_env_cfg:FactoryTaskGearMeshCfg",
            rl_games_cfg_entry_point="isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
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
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg:PickPlaceGR1T2EnvCfg",
            robomimic_bc_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.agents:robomimic/bc_rnn_low_dim.json",
        ),
    },
    {
        "id": "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_pink_ik_env_cfg:NutPourGR1T2PinkIKEnvCfg",
            robomimic_bc_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.agents:robomimic/bc_rnn_image_nut_pouring.json",
        ),
    },
    {
        "id": "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_pink_ik_env_cfg:ExhaustPipeGR1T2PinkIKEnvCfg",
            robomimic_bc_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.agents:robomimic/bc_rnn_image_exhaust_pipe.json",
        ),
    },
    {
        "id": "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_waist_enabled_env_cfg:PickPlaceGR1T2WaistEnabledEnvCfg",
            robomimic_bc_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.agents:robomimic/bc_rnn_low_dim.json",
        ),
    },
    {
        "id": "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": _isaacsim_only_kwargs(
            env_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg:PickPlaceG1InspireFTPEnvCfg",
            robomimic_bc_cfg_entry_point="isaaclab_tasks.manager_based.manipulation.pick_place.agents:robomimic/bc_rnn_low_dim.json",
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
