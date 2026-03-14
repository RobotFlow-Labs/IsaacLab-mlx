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


@configclass
class MacAnymalCFlatEnvCfg:
    """Configuration aligned with the upstream flat ANYmal-C locomotion task where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 200.0
    decimation: int = 4
    episode_length_s: float = 20.0
    action_space: int = 12
    observation_space: int = 48
    state_space: int = 0

    env_spacing: float = 4.0
    terrain_tile_size: tuple[float, float] = (4.0, 4.0)
    default_root_height: float = 0.55
    min_root_height: float = 0.18
    nominal_leg_extension: float = 0.52
    foot_clearance: float = 0.08
    gait_frequency: float = 1.6

    action_scale: float = 0.5
    command_scale: float = 1.0
    joint_reset_noise: float = 0.05
    default_joint_pos: tuple[float, ...] = (
        0.0,
        0.6,
        -1.2,
        0.0,
        0.6,
        -1.2,
        0.0,
        0.6,
        -1.2,
        0.0,
        0.6,
        -1.2,
    )

    joint_stiffness: float = 28.0
    joint_damping: float = 4.5
    joint_inertia: float = 1.6
    command_tracking_gain: float = 4.0
    yaw_tracking_gain: float = 4.5
    root_lin_damping: float = 2.2
    root_ang_damping: float = 2.0
    height_stiffness: float = 36.0
    height_damping: float = 7.5
    balance_gain: float = 2.5
    orientation_height_penalty: float = 4.0

    contact_history_length: int = 3
    contact_margin: float = 0.02
    contact_stiffness: float = 2200.0
    contact_damping: float = 60.0
    contact_force_threshold: float = 1.0

    lin_vel_reward_scale: float = 1.0
    yaw_rate_reward_scale: float = 0.5
    z_vel_reward_scale: float = -2.0
    ang_vel_reward_scale: float = -0.05
    joint_torque_reward_scale: float = -2.5e-5
    joint_accel_reward_scale: float = -2.5e-7
    action_rate_reward_scale: float = -0.01
    feet_air_time_reward_scale: float = 0.5
    undesired_contact_reward_scale: float = -1.0
    flat_orientation_reward_scale: float = -5.0

    seed: int = 42
