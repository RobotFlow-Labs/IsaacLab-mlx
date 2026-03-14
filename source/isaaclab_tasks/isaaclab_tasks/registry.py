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
        "id": "Isaac-Cartpole-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacCartpoleEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim:MacCartpoleEnvCfg",
        },
    },
    {
        "id": "Isaac-Cart-Double-Pendulum-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacCartDoublePendulumEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim:MacCartDoublePendulumEnvCfg",
        },
    },
    {
        "id": "Isaac-Quadcopter-Direct-v0",
        "entry_point": "isaaclab.backends.mac_sim:MacQuadcopterEnv",
        "kwargs": {
            "env_cfg_entry_point": "isaaclab.backends.mac_sim:MacQuadcopterEnvCfg",
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
]
