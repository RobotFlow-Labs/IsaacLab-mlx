# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""macOS-native simulator backends and runnable reference environments."""

from .env_cfgs import MacCartDoublePendulumEnvCfg, MacCartpoleEnvCfg, MacQuadcopterEnvCfg
from .state_primitives import BatchedArticulationState, BatchedRootState, EnvironmentOriginGrid
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
