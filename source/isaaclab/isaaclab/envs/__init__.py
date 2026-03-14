# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for environment definitions."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_MODULE_EXPORTS = {
    "VecEnvObs": (".common", "VecEnvObs"),
    "VecEnvStepReturn": (".common", "VecEnvStepReturn"),
    "ViewerCfg": (".common", "ViewerCfg"),
    "DirectMARLEnv": (".direct_marl_env", "DirectMARLEnv"),
    "DirectMARLEnvCfg": (".direct_marl_env_cfg", "DirectMARLEnvCfg"),
    "DirectRLEnv": (".direct_rl_env", "DirectRLEnv"),
    "DirectRLEnvCfg": (".direct_rl_env_cfg", "DirectRLEnvCfg"),
    "ManagerBasedEnv": (".manager_based_env", "ManagerBasedEnv"),
    "ManagerBasedEnvCfg": (".manager_based_env_cfg", "ManagerBasedEnvCfg"),
    "ManagerBasedRLEnv": (".manager_based_rl_env", "ManagerBasedRLEnv"),
    "ManagerBasedRLEnvCfg": (".manager_based_rl_env_cfg", "ManagerBasedRLEnvCfg"),
    "ManagerBasedRLMimicEnv": (".manager_based_rl_mimic_env", "ManagerBasedRLMimicEnv"),
    "DataGenConfig": (".mimic_env_cfg", "DataGenConfig"),
    "SubTaskConfig": (".mimic_env_cfg", "SubTaskConfig"),
    "SubTaskConstraintType": (".mimic_env_cfg", "SubTaskConstraintType"),
    "SubTaskConstraintCoordinationScheme": (".mimic_env_cfg", "SubTaskConstraintCoordinationScheme"),
    "SubTaskConstraintConfig": (".mimic_env_cfg", "SubTaskConstraintConfig"),
    "multi_agent_to_single_agent": (".utils.marl", "multi_agent_to_single_agent"),
    "multi_agent_with_one_agent": (".utils.marl", "multi_agent_with_one_agent"),
}
_ISAACSIM_ONLY_EXPORTS = {
    "DirectMARLEnvCfg",
    "DirectMARLEnv",
    "DirectRLEnvCfg",
    "DirectRLEnv",
    "ManagerBasedEnvCfg",
    "ManagerBasedEnv",
    "ManagerBasedRLEnvCfg",
    "ManagerBasedRLEnv",
    "ManagerBasedRLMimicEnv",
    "multi_agent_to_single_agent",
    "multi_agent_with_one_agent",
}

__all__ = ["mdp", "ui", *_MODULE_EXPORTS.keys()]


def __getattr__(name: str):
    if name in {"mdp", "ui"}:
        if current_runtime().sim_backend != "isaacsim":
            raise UnsupportedBackendError(
                f"`isaaclab.envs.{name}` currently requires `sim-backend=isaacsim`."
                " The `mac-sim` path will expose separate environment modules."
            )
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in _ISAACSIM_ONLY_EXPORTS and current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.envs.{name}` currently requires `sim-backend=isaacsim`."
            " Use configuration objects or the forthcoming macOS-native environment bases for `mac-sim`."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
