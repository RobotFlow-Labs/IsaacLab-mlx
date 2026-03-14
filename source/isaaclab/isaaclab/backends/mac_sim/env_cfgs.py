# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-only MLX/mac task surfaces used by the lazy task registry."""

from __future__ import annotations

import math
from typing import Any

from isaaclab.utils.configclass import configclass


@configclass
class MacCartpoleEnvCfg:
    """Cartpole configuration aligned with the upstream IsaacLab direct task where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 5.0

    action_scale: float = 100.0
    action_space: Any = 1
    observation_space: Any = 4
    state_space: Any = 0

    max_cart_pos: float = 3.0
    initial_pole_angle_range: tuple[float, float] = (-0.25 * math.pi, 0.25 * math.pi)

    rew_scale_alive: float = 1.0
    rew_scale_terminated: float = -2.0
    rew_scale_pole_pos: float = -1.0
    rew_scale_cart_vel: float = -0.01
    rew_scale_pole_vel: float = -0.005

    gravity: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_half_length: float = 0.5
    force_mag: float = 1.0

    seed: int = 42


@configclass
class MacCartDoublePendulumEnvCfg:
    """Configuration aligned with upstream cart-double-pendulum semantics where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 5.0
    possible_agents: tuple[str, str] = ("cart", "pendulum")
    action_spaces: dict[str, int] = {"cart": 1, "pendulum": 1}
    observation_spaces: dict[str, int] = {"cart": 4, "pendulum": 3}
    state_space: int = -1

    max_cart_pos: float = 3.0
    initial_pole_angle_range: tuple[float, float] = (-0.25, 0.25)
    initial_pendulum_angle_range: tuple[float, float] = (-0.25, 0.25)

    cart_action_scale: float = 100.0
    pendulum_action_scale: float = 50.0

    rew_scale_alive: float = 1.0
    rew_scale_terminated: float = -2.0
    rew_scale_cart_pos: float = 0.0
    rew_scale_cart_vel: float = -0.01
    rew_scale_pole_pos: float = -1.0
    rew_scale_pole_vel: float = -0.01
    rew_scale_pendulum_pos: float = -1.0
    rew_scale_pendulum_vel: float = -0.01

    gravity: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    mass_pendulum: float = 0.08
    pole_half_length: float = 0.5
    pendulum_half_length: float = 0.45
    cart_force_scale: float = 1.0
    pendulum_torque_scale: float = 1.0
    pendulum_damping: float = 0.05

    seed: int = 42


@configclass
class MacQuadcopterEnvCfg:
    """Configuration aligned with the upstream quadcopter task where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 100.0
    decimation: int = 2
    episode_length_s: float = 10.0
    action_space: int = 4
    observation_space: int = 12
    state_space: int = 0

    env_spacing: float = 2.5
    mass: float = 0.032
    gravity: float = 9.81
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    angular_damping: float = 0.08
    linear_damping_xy: float = 0.05
    linear_damping_z: float = 0.03
    lateral_accel_scale: float = 2.5

    min_height: float = 0.1
    max_height: float = 2.0

    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0

    seed: int = 42
