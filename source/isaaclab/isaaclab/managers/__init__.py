# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for environment managers, lazily loaded by backend capability."""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_MODULE_EXPORTS = {
    "ActionManager": (".action_manager", "ActionManager"),
    "ActionTerm": (".action_manager", "ActionTerm"),
    "CommandManager": (".command_manager", "CommandManager"),
    "CommandTerm": (".command_manager", "CommandTerm"),
    "CurriculumManager": (".curriculum_manager", "CurriculumManager"),
    "EventManager": (".event_manager", "EventManager"),
    "ManagerBase": (".manager_base", "ManagerBase"),
    "ManagerTermBase": (".manager_base", "ManagerTermBase"),
    "ActionTermCfg": (".manager_term_cfg", "ActionTermCfg"),
    "CommandTermCfg": (".manager_term_cfg", "CommandTermCfg"),
    "CurriculumTermCfg": (".manager_term_cfg", "CurriculumTermCfg"),
    "EventTermCfg": (".manager_term_cfg", "EventTermCfg"),
    "ManagerTermBaseCfg": (".manager_term_cfg", "ManagerTermBaseCfg"),
    "ObservationGroupCfg": (".manager_term_cfg", "ObservationGroupCfg"),
    "ObservationTermCfg": (".manager_term_cfg", "ObservationTermCfg"),
    "RecorderTermCfg": (".manager_term_cfg", "RecorderTermCfg"),
    "RewardTermCfg": (".manager_term_cfg", "RewardTermCfg"),
    "TerminationTermCfg": (".manager_term_cfg", "TerminationTermCfg"),
    "ObservationManager": (".observation_manager", "ObservationManager"),
    "DatasetExportMode": (".recorder_manager", "DatasetExportMode"),
    "RecorderManager": (".recorder_manager", "RecorderManager"),
    "RecorderManagerBaseCfg": (".recorder_manager", "RecorderManagerBaseCfg"),
    "RecorderTerm": (".recorder_manager", "RecorderTerm"),
    "RewardManager": (".reward_manager", "RewardManager"),
    "SceneEntityCfg": (".scene_entity_cfg", "SceneEntityCfg"),
    "TerminationManager": (".termination_manager", "TerminationManager"),
}

__all__ = [*_MODULE_EXPORTS.keys()]


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.managers.{name}` currently requires `sim-backend=isaacsim`."
            " Manager interfaces for `mac-sim` are exposed progressively via backend capabilities."
        )

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
