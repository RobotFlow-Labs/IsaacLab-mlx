# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-only MLX/mac task surfaces used by the lazy task registry."""

from __future__ import annotations

import math
from typing import Any

from isaaclab.utils.configclass import configclass

ROUGH_HEIGHT_SCAN_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.35, -0.35),
    (-0.35, 0.0),
    (-0.35, 0.35),
    (0.0, -0.35),
    (0.0, 0.0),
    (0.0, 0.35),
    (0.35, -0.35),
    (0.35, 0.0),
    (0.35, 0.35),
)

H1_FLAT_OBSERVATION_SPACE = 69


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
class MacCartpoleRGBCameraEnvCfg(MacCartpoleEnvCfg):
    """Synthetic RGB camera cartpole configuration for the mac-native sensor path."""

    camera_mode: str = "rgb"
    image_height: int = 100
    image_width: int = 100
    camera_channels: int = 3
    observation_space: Any = [100, 100, 3]
    camera_max_depth_m: float = 6.0
    initial_pole_angle_range: tuple[float, float] = (-0.125 * math.pi, 0.125 * math.pi)


@configclass
class MacCartpoleDepthCameraEnvCfg(MacCartpoleRGBCameraEnvCfg):
    """Synthetic depth camera cartpole configuration for the mac-native sensor path."""

    camera_mode: str = "depth"
    camera_channels: int = 1
    observation_space: Any = [100, 100, 1]


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
    terrain_type: str = "plane"
    terrain_tile_size: tuple[float, float] = (4.0, 4.0)
    terrain_border_width: float = 0.0
    terrain_height_amplitude: float = 0.0
    terrain_wavelength: tuple[float, float] = (1.0, 1.0)
    default_root_height: float = 0.55
    min_root_height: float = 0.18
    height_scan_enabled: bool = False
    height_scan_offsets: tuple[tuple[float, float], ...] = ()
    height_scan_max_distance: float = 2.0
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


@configclass
class MacAnymalCRoughEnvCfg(MacAnymalCFlatEnvCfg):
    """Configuration aligned with the upstream rough ANYmal-C task at a semantic level."""

    terrain_type: str = "wave"
    terrain_tile_size: tuple[float, float] = (5.0, 5.0)
    terrain_border_width: float = 0.15
    terrain_height_amplitude: float = 0.06
    terrain_wavelength: tuple[float, float] = (1.4, 1.0)
    default_root_height: float = 0.62
    min_root_height: float = 0.22
    height_scan_enabled: bool = True
    height_scan_offsets: tuple[tuple[float, float], ...] = ROUGH_HEIGHT_SCAN_OFFSETS
    height_scan_max_distance: float = 2.5
    lin_vel_reward_scale: float = 1.15
    yaw_rate_reward_scale: float = 0.65
    z_vel_reward_scale: float = -2.5
    flat_orientation_reward_scale: float = -6.0


@configclass
class MacH1FlatEnvCfg:
    """Configuration aligned with the upstream flat H1 locomotion task where practical."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 200.0
    decimation: int = 4
    episode_length_s: float = 20.0
    action_space: int = 19
    observation_space: int = H1_FLAT_OBSERVATION_SPACE
    state_space: int = 0

    env_spacing: float = 4.0
    terrain_type: str = "plane"
    terrain_tile_size: tuple[float, float] = (4.0, 4.0)
    terrain_border_width: float = 0.0
    terrain_height_amplitude: float = 0.0
    terrain_wavelength: tuple[float, float] = (1.0, 1.0)
    default_root_height: float = 1.05
    min_root_height: float = 0.55
    height_scan_enabled: bool = False
    height_scan_offsets: tuple[tuple[float, float], ...] = ()
    height_scan_max_distance: float = 2.0
    nominal_leg_extension: float = 0.86
    foot_clearance: float = 0.12
    gait_frequency: float = 1.35

    action_scale: float = 0.35
    command_x_range: tuple[float, float] = (0.0, 1.0)
    yaw_command_scale: float = 1.0
    joint_reset_noise: float = 0.04
    default_joint_pos: tuple[float, ...] = (
        0.0,
        0.0,
        -0.28,
        0.79,
        -0.52,
        0.0,
        0.0,
        -0.28,
        0.79,
        -0.52,
        0.0,
        0.28,
        0.0,
        0.0,
        0.52,
        0.28,
        0.0,
        0.0,
        0.52,
    )

    leg_joint_stiffness: float = 40.0
    leg_joint_damping: float = 5.0
    leg_joint_inertia: float = 1.9
    torso_joint_stiffness: float = 18.0
    torso_joint_damping: float = 3.0
    torso_joint_inertia: float = 1.4
    arm_joint_stiffness: float = 12.0
    arm_joint_damping: float = 2.5
    arm_joint_inertia: float = 1.1
    joint_position_limit: float = 1.2
    ankle_soft_limit: float = 0.9

    command_tracking_gain: float = 3.6
    yaw_tracking_gain: float = 4.0
    root_lin_damping: float = 2.4
    root_ang_damping: float = 2.1
    height_stiffness: float = 34.0
    height_damping: float = 7.0
    balance_gain: float = 2.8
    orientation_height_penalty: float = 4.5

    contact_history_length: int = 3
    contact_margin: float = 0.02
    contact_stiffness: float = 2200.0
    contact_damping: float = 60.0
    contact_force_threshold: float = 1.0

    lin_vel_reward_scale: float = 1.0
    yaw_rate_reward_scale: float = 1.0
    feet_air_time_reward_scale: float = 1.0
    feet_slide_reward_scale: float = -0.25
    ankle_limit_reward_scale: float = -1.0
    joint_deviation_hip_reward_scale: float = -0.2
    joint_deviation_arms_reward_scale: float = -0.2
    joint_deviation_torso_reward_scale: float = -0.1
    flat_orientation_reward_scale: float = -1.0
    action_rate_reward_scale: float = -0.005
    joint_accel_reward_scale: float = -1.25e-7

    seed: int = 42


@configclass
class MacH1RoughEnvCfg(MacH1FlatEnvCfg):
    """Configuration aligned with the upstream rough H1 locomotion task at a semantic level."""

    terrain_type: str = "wave"
    terrain_tile_size: tuple[float, float] = (5.0, 5.0)
    terrain_border_width: float = 0.15
    terrain_height_amplitude: float = 0.055
    terrain_wavelength: tuple[float, float] = (1.5, 1.15)
    default_root_height: float = 1.08
    min_root_height: float = 0.62
    observation_space: int = H1_FLAT_OBSERVATION_SPACE + len(ROUGH_HEIGHT_SCAN_OFFSETS)
    height_scan_enabled: bool = True
    height_scan_offsets: tuple[tuple[float, float], ...] = ROUGH_HEIGHT_SCAN_OFFSETS
    height_scan_max_distance: float = 2.5
    lin_vel_reward_scale: float = 1.1
    yaw_rate_reward_scale: float = 1.05
    flat_orientation_reward_scale: float = -1.4
    feet_slide_reward_scale: float = -0.3
    ankle_limit_reward_scale: float = -1.2


@configclass
class MacFrankaReachEnvCfg:
    """Configuration for the first mac-native Franka reach slice."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 6.0
    action_space: int = 7
    observation_space: int = 23
    state_space: int = 0

    action_scale: float = 0.35
    action_rate_penalty_scale: float = -0.01
    joint_vel_penalty_scale: float = -0.0005
    reach_reward_scale: float = 1.25
    success_bonus: float = 2.0
    success_threshold: float = 0.06
    distance_reward_gain: float = 7.5

    joint_stiffness: float = 14.0
    joint_damping: float = 3.0
    joint_inertia: float = 1.1
    joint_reset_noise: float = 0.05
    default_joint_pos: tuple[float, ...] = (0.0, -0.55, 0.0, -2.1, 0.0, 1.65, 0.75)
    joint_lower_limits: tuple[float, ...] = (-2.6, -1.8, -2.6, -3.0, -2.6, -0.1, -2.6)
    joint_upper_limits: tuple[float, ...] = (2.6, 1.8, 2.6, -0.05, 2.6, 3.5, 2.6)

    target_x_range: tuple[float, float] = (0.42, 0.72)
    target_y_range: tuple[float, float] = (-0.24, 0.24)
    target_z_range: tuple[float, float] = (0.18, 0.42)

    seed: int = 42


@configclass
class MacFrankaLiftEnvCfg(MacFrankaReachEnvCfg):
    """Configuration for the first mac-native Franka cube-lift slice."""

    action_space: int = 8
    observation_space: int = 27
    episode_length_s: float = 8.0

    action_scale: float = 0.28
    gripper_action_scale: float = 0.03
    default_joint_pos: tuple[float, ...] = (0.0, -0.55, 0.0, -2.1, 0.0, 1.65, 0.75, 0.04)
    joint_lower_limits: tuple[float, ...] = (-2.6, -1.8, -2.6, -3.0, -2.6, -0.1, -2.6, 0.0)
    joint_upper_limits: tuple[float, ...] = (2.6, 1.8, 2.6, -0.05, 2.6, 3.5, 2.6, 0.08)

    cube_x_range: tuple[float, float] = (0.48, 0.66)
    cube_y_range: tuple[float, float] = (-0.12, 0.12)
    table_height: float = 0.04
    grasp_distance_threshold: float = 0.07
    gripper_closed_threshold: float = 0.03
    grasp_offset_z: float = 0.055
    lift_success_height: float = 0.20

    grasp_reward_scale: float = 0.5
    lift_reward_scale: float = 4.0
    lift_success_bonus: float = 4.0


@configclass
class MacFrankaStackEnvCfg(MacFrankaLiftEnvCfg):
    """Configuration for a reduced mac-native Franka cube-stacking slice."""

    observation_space: int = 33
    episode_length_s: float = 10.0

    support_cube_x_range: tuple[float, float] = (0.44, 0.58)
    support_cube_y_range: tuple[float, float] = (-0.06, 0.06)
    movable_cube_offset_x_range: tuple[float, float] = (0.10, 0.18)
    movable_cube_offset_y_range: tuple[float, float] = (0.06, 0.14)
    stack_offset_z: float = 0.04
    stack_xy_threshold: float = 0.04
    stack_z_threshold: float = 0.03
    stack_release_open_threshold: float = 0.05

    grasp_reward_scale: float = 0.45
    lift_reward_scale: float = 1.25
    stack_align_reward_scale: float = 2.5
    stack_distance_reward_gain: float = 9.0
    stack_success_bonus: float = 6.0


@configclass
class MacFrankaCabinetEnvCfg(MacFrankaLiftEnvCfg):
    """Configuration for a reduced mac-native Franka cabinet-drawer slice."""

    observation_space: int = 28
    episode_length_s: float = 10.0

    handle_anchor_x_range: tuple[float, float] = (0.48, 0.58)
    handle_anchor_y_range: tuple[float, float] = (-0.05, 0.05)
    handle_anchor_z_range: tuple[float, float] = (0.16, 0.22)
    handle_grasp_threshold: float = 0.065
    drawer_open_distance_max: float = 0.24
    drawer_success_distance: float = 0.18

    grasp_reward_scale: float = 0.45
    open_reward_scale: float = 5.0
    drawer_success_bonus: float = 6.0


@configclass
class MacFrankaStackRgbEnvCfg(MacFrankaStackEnvCfg):
    """Configuration for a reduced three-cube Franka stack slice."""

    observation_space: int = 42
    episode_length_s: float = 12.0

    support_cube_x_range: tuple[float, float] = (0.44, 0.56)
    support_cube_y_range: tuple[float, float] = (-0.05, 0.05)
    middle_cube_offset_x_range: tuple[float, float] = (0.10, 0.16)
    middle_cube_offset_y_range: tuple[float, float] = (0.08, 0.14)
    top_cube_offset_x_range: tuple[float, float] = (-0.16, -0.10)
    top_cube_offset_y_range: tuple[float, float] = (0.08, 0.14)

    middle_stage_bonus: float = 2.5
    top_stack_align_reward_scale: float = 3.0
    top_stack_distance_reward_gain: float = 10.0
    stack_success_bonus: float = 8.0
