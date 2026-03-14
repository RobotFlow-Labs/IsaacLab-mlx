# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""macOS-native simulator backends and runnable reference environments."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "BatchedContactSensorState": (".contacts", "BatchedContactSensorState"),
    "mac_env_diagnostics": (".diagnostics", "mac_env_diagnostics"),
    "MacAnymalCFlatEnvCfg": (".env_cfgs", "MacAnymalCFlatEnvCfg"),
    "MacCartDoublePendulumEnvCfg": (".env_cfgs", "MacCartDoublePendulumEnvCfg"),
    "MacCartpoleEnvCfg": (".env_cfgs", "MacCartpoleEnvCfg"),
    "MacH1FlatEnvCfg": (".env_cfgs", "MacH1FlatEnvCfg"),
    "MacQuadcopterEnvCfg": (".env_cfgs", "MacQuadcopterEnvCfg"),
    "build_checkpoint_metadata": (".ppo_training", "build_checkpoint_metadata"),
    "checkpoint_metadata_path": (".ppo_training", "checkpoint_metadata_path"),
    "compute_gae": (".ppo_training", "compute_gae"),
    "mean_recent_return": (".ppo_training", "mean_recent_return"),
    "normalize_advantages": (".ppo_training", "normalize_advantages"),
    "play_categorical_policy_checkpoint": (".ppo_training", "play_categorical_policy_checkpoint"),
    "play_gaussian_policy_checkpoint": (".ppo_training", "play_gaussian_policy_checkpoint"),
    "read_checkpoint_metadata": (".ppo_training", "read_checkpoint_metadata"),
    "resolve_resume_hidden_dim": (".ppo_training", "resolve_resume_hidden_dim"),
    "save_policy_checkpoint": (".ppo_training", "save_policy_checkpoint"),
    "write_checkpoint_metadata": (".ppo_training", "write_checkpoint_metadata"),
    "action_rate_l2": (".locomotion", "action_rate_l2"),
    "base_contact_termination": (".locomotion", "base_contact_termination"),
    "feet_air_time_reward": (".locomotion", "feet_air_time_reward"),
    "flat_orientation_l2": (".locomotion", "flat_orientation_l2"),
    "terrain_out_of_bounds": (".locomotion", "terrain_out_of_bounds"),
    "track_linear_velocity_xy_exp": (".locomotion", "track_linear_velocity_xy_exp"),
    "track_yaw_rate_z_exp": (".locomotion", "track_yaw_rate_z_exp"),
    "undesired_contacts": (".locomotion", "undesired_contacts"),
    "DeterministicResetSampler": (".reset_primitives", "DeterministicResetSampler"),
    "DEFAULT_HEIGHT_SCAN_OFFSETS": (".sensors", "DEFAULT_HEIGHT_SCAN_OFFSETS"),
    "MacPlaneRaycastSensor": (".sensors", "MacPlaneRaycastSensor"),
    "RolloutTrace": (".rollout", "RolloutTrace"),
    "replay_actions": (".rollout", "replay_actions"),
    "rollout_env": (".rollout", "rollout_env"),
    "BatchedArticulationState": (".state_primitives", "BatchedArticulationState"),
    "BatchedRootState": (".state_primitives", "BatchedRootState"),
    "EnvironmentOriginGrid": (".state_primitives", "EnvironmentOriginGrid"),
    "MacPlaneTerrain": (".terrain", "MacPlaneTerrain"),
    "MacAnymalCFlatEnv": (".anymal_c", "MacAnymalCFlatEnv"),
    "MacAnymalCFlatSimBackend": (".anymal_c", "MacAnymalCFlatSimBackend"),
    "MacAnymalCPolicy": (".anymal_c", "MacAnymalCPolicy"),
    "MacAnymalCTrainCfg": (".anymal_c", "MacAnymalCTrainCfg"),
    "play_anymal_c_policy": (".anymal_c", "play_anymal_c_policy"),
    "train_anymal_c_policy": (".anymal_c", "train_anymal_c_policy"),
    "MacH1FlatEnv": (".h1", "MacH1FlatEnv"),
    "MacH1FlatSimBackend": (".h1", "MacH1FlatSimBackend"),
    "MacH1Policy": (".h1", "MacH1Policy"),
    "MacH1TrainCfg": (".h1", "MacH1TrainCfg"),
    "play_h1_policy": (".h1", "play_h1_policy"),
    "train_h1_policy": (".h1", "train_h1_policy"),
    "MacCartpoleEnv": (".cartpole", "MacCartpoleEnv"),
    "MacCartpolePolicy": (".cartpole", "MacCartpolePolicy"),
    "MacCartpoleSimBackend": (".cartpole", "MacCartpoleSimBackend"),
    "MacCartpoleTrainCfg": (".cartpole", "MacCartpoleTrainCfg"),
    "play_cartpole_policy": (".cartpole", "play_cartpole_policy"),
    "train_cartpole_policy": (".cartpole", "train_cartpole_policy"),
    "MacCartDoublePendulumEnv": (".cart_double_pendulum", "MacCartDoublePendulumEnv"),
    "MacCartDoublePendulumSimBackend": (".cart_double_pendulum", "MacCartDoublePendulumSimBackend"),
    "MacQuadcopterEnv": (".quadcopter", "MacQuadcopterEnv"),
    "MacQuadcopterSimBackend": (".quadcopter", "MacQuadcopterSimBackend"),
    "SHOWCASE_CFGS": (".showcase", "SHOWCASE_CFGS"),
    "BoxBoxEnvCfg": (".showcase", "BoxBoxEnvCfg"),
    "BoxDiscreteEnvCfg": (".showcase", "BoxDiscreteEnvCfg"),
    "BoxMultiDiscreteEnvCfg": (".showcase", "BoxMultiDiscreteEnvCfg"),
    "DictBoxEnvCfg": (".showcase", "DictBoxEnvCfg"),
    "DictDiscreteEnvCfg": (".showcase", "DictDiscreteEnvCfg"),
    "DictMultiDiscreteEnvCfg": (".showcase", "DictMultiDiscreteEnvCfg"),
    "DiscreteBoxEnvCfg": (".showcase", "DiscreteBoxEnvCfg"),
    "DiscreteDiscreteEnvCfg": (".showcase", "DiscreteDiscreteEnvCfg"),
    "DiscreteMultiDiscreteEnvCfg": (".showcase", "DiscreteMultiDiscreteEnvCfg"),
    "MacCartpoleShowcaseEnv": (".showcase", "MacCartpoleShowcaseEnv"),
    "MultiDiscreteBoxEnvCfg": (".showcase", "MultiDiscreteBoxEnvCfg"),
    "MultiDiscreteDiscreteEnvCfg": (".showcase", "MultiDiscreteDiscreteEnvCfg"),
    "MultiDiscreteMultiDiscreteEnvCfg": (".showcase", "MultiDiscreteMultiDiscreteEnvCfg"),
    "TupleBoxEnvCfg": (".showcase", "TupleBoxEnvCfg"),
    "TupleDiscreteEnvCfg": (".showcase", "TupleDiscreteEnvCfg"),
    "TupleMultiDiscreteEnvCfg": (".showcase", "TupleMultiDiscreteEnvCfg"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
