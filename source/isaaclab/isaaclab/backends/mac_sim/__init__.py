# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""macOS-native simulator backends and runnable reference environments."""

from .contacts import BatchedContactSensorState
from .diagnostics import mac_env_diagnostics
from .env_cfgs import MacCartDoublePendulumEnvCfg, MacCartpoleEnvCfg, MacQuadcopterEnvCfg
from .locomotion import (
    action_rate_l2,
    base_contact_termination,
    feet_air_time_reward,
    flat_orientation_l2,
    terrain_out_of_bounds,
    track_linear_velocity_xy_exp,
    track_yaw_rate_z_exp,
    undesired_contacts,
)
from .reset_primitives import DeterministicResetSampler
from .rollout import RolloutTrace, replay_actions, rollout_env
from .state_primitives import BatchedArticulationState, BatchedRootState, EnvironmentOriginGrid
from .terrain import MacPlaneTerrain
from .cartpole import (
    MacCartpoleEnv,
    MacCartpolePolicy,
    MacCartpoleSimBackend,
    MacCartpoleTrainCfg,
    play_cartpole_policy,
    train_cartpole_policy,
)
from .cart_double_pendulum import (
    MacCartDoublePendulumEnv,
    MacCartDoublePendulumSimBackend,
)
from .quadcopter import (
    MacQuadcopterEnv,
    MacQuadcopterSimBackend,
)
from .showcase import (
    SHOWCASE_CFGS,
    BoxBoxEnvCfg,
    BoxDiscreteEnvCfg,
    BoxMultiDiscreteEnvCfg,
    DictBoxEnvCfg,
    DictDiscreteEnvCfg,
    DictMultiDiscreteEnvCfg,
    DiscreteBoxEnvCfg,
    DiscreteDiscreteEnvCfg,
    DiscreteMultiDiscreteEnvCfg,
    MacCartpoleShowcaseEnv,
    MultiDiscreteBoxEnvCfg,
    MultiDiscreteDiscreteEnvCfg,
    MultiDiscreteMultiDiscreteEnvCfg,
    TupleBoxEnvCfg,
    TupleDiscreteEnvCfg,
    TupleMultiDiscreteEnvCfg,
)
