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
class MacUR10eDeployReachEnvCfg:
    """Reduced mac-native UR10e deploy-reach configuration."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 12.0
    action_space: int = 6
    observation_space: int = 19
    state_space: int = 0

    action_scale: float = 0.0625
    action_rate_penalty_scale: float = -0.005
    joint_vel_penalty_scale: float = -0.0005
    reach_reward_scale: float = 1.5
    orientation_reward_scale: float = 0.8
    success_bonus: float = 2.5
    success_position_threshold: float = 0.06
    success_orientation_threshold: float = 0.18
    position_reward_gain: float = 5.5
    orientation_reward_gain: float = 8.0

    joint_stiffness: float = 12.0
    joint_damping: float = 3.0
    joint_inertia: float = 1.15
    joint_reset_noise: float = 0.125
    default_joint_pos: tuple[float, ...] = (0.0, -1.35, 1.55, -1.75, -1.57, 0.0)
    joint_lower_limits: tuple[float, ...] = (-2.0 * math.pi, -math.pi, -math.pi, -2.0 * math.pi, -2.0 * math.pi, -2.0 * math.pi)
    joint_upper_limits: tuple[float, ...] = (2.0 * math.pi, 0.0, math.pi, 2.0 * math.pi, 2.0 * math.pi, 2.0 * math.pi)

    target_x_range: tuple[float, float] = (0.6375, 1.1375)
    target_y_range: tuple[float, float] = (-0.35, -0.10)
    target_z_range: tuple[float, float] = (0.10, 0.30)
    target_roll_range: tuple[float, float] = (math.pi - math.pi / 6.0, math.pi + math.pi / 6.0)
    target_pitch_range: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0)
    target_yaw_range: tuple[float, float] = (-math.pi / 2.0 - 2.0 * math.pi / 3.0, -math.pi / 2.0 + 2.0 * math.pi / 3.0)

    task_name: str = "ur10e-deploy-reach"
    semantic_contract: str = "reduced-analytic-pose"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "This mac-native slice preserves the UR10e deploy-reach joint-space pose workflow with "
        "analytic pose tracking instead of the full deployment frame-transform and ROS inference stack."
    )
    seed: int = 42


@configclass
class MacUR10ReachEnvCfg(MacUR10eDeployReachEnvCfg):
    """Reduced mac-native UR10 reach configuration."""

    episode_length_s: float = 8.0
    action_scale: float = 0.075
    success_bonus: float = 2.0
    target_x_range: tuple[float, float] = (0.45, 0.95)
    target_y_range: tuple[float, float] = (-0.28, 0.28)
    target_z_range: tuple[float, float] = (0.16, 0.42)
    target_roll_range: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0)
    target_pitch_range: tuple[float, float] = (math.pi / 2.0, math.pi / 2.0)
    target_yaw_range: tuple[float, float] = (-math.pi / 2.0, math.pi / 2.0)
    task_name: str = "ur10-reach"
    semantic_contract: str = "reduced-analytic-pose"
    upstream_alias_semantics_preserved: bool = False


@configclass
class MacUR10eDeployReachRosInferenceEnvCfg(MacUR10eDeployReachEnvCfg):
    """Reduced mac-native UR10e deploy-reach configuration for the ROS-inference variant."""

    task_name: str = "ur10e-deploy-reach"
    semantic_contract: str = "reduced-no-ros-inference"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native UR10e deploy-reach slice preserves the joint-space reach workflow, but it "
        "does not include the upstream ROS inference transport or deployed-robot runtime stack."
    )


@configclass
class MacUR10eGearAssembly2F140EnvCfg(MacUR10eDeployReachEnvCfg):
    """Reduced mac-native UR10e gear-assembly configuration for the Robotiq 2F-140 gripper."""

    episode_length_s: float = 10.0
    reach_reward_scale: float = 1.25
    orientation_reward_scale: float = 0.9
    success_bonus: float = 3.0
    target_x_range: tuple[float, float] = (0.90, 1.06)
    target_y_range: tuple[float, float] = (-0.26, -0.14)
    target_z_range: tuple[float, float] = (0.02, 0.10)
    target_roll_range: tuple[float, float] = (0.0, 0.0)
    target_pitch_range: tuple[float, float] = (math.pi / 2.0, math.pi / 2.0)
    target_yaw_range: tuple[float, float] = (-math.pi / 2.0, -math.pi / 2.0)

    alignment_position_threshold: float = 0.055
    alignment_orientation_threshold: float = 0.2
    insertion_depth_max: float = 0.065
    insertion_success_depth: float = 0.052
    insertion_rate: float = 0.18
    insertion_decay_rate: float = 0.08
    insertion_reward_scale: float = 1.1
    insertion_reward_gain: float = 6.0
    gear_type_offsets_x: tuple[float, float, float] = (0.076125, 0.030375, -0.045375)
    task_name: str = "ur10e-gear-assembly-2f140"
    semantic_contract: str = "reduced-analytic-assembly"
    upstream_alias_semantics_preserved: bool = False
    gripper_variant: str = "2f140"
    contract_notes: str = (
        "This mac-native slice preserves the UR10e gear-assembly pose-command workflow with "
        "analytic shaft alignment and scalar insertion progress rather than the full contact-rich "
        "factory gear dynamics, gripper compliance, and ROS deployment stack."
    )


@configclass
class MacUR10eGearAssembly2F85EnvCfg(MacUR10eGearAssembly2F140EnvCfg):
    """Reduced mac-native UR10e gear-assembly configuration for the Robotiq 2F-85 gripper."""

    target_x_range: tuple[float, float] = (0.88, 1.04)
    target_z_range: tuple[float, float] = (0.03, 0.11)
    insertion_depth_max: float = 0.055
    insertion_success_depth: float = 0.044
    insertion_rate: float = 0.16
    insertion_reward_scale: float = 1.0
    task_name: str = "ur10e-gear-assembly-2f85"
    gripper_variant: str = "2f85"
    contract_notes: str = (
        "This mac-native slice preserves the UR10e gear-assembly pose-command workflow with "
        "analytic shaft alignment and scalar insertion progress for the 2F-85 gripper instead of "
        "the full factory contact dynamics, gripper compliance, and ROS deployment stack."
    )


@configclass
class MacUR10eGearAssembly2F140RosInferenceEnvCfg(MacUR10eGearAssembly2F140EnvCfg):
    """Reduced mac-native 2F-140 gear-assembly configuration for the ROS-inference variant."""

    semantic_contract: str = "reduced-no-ros-inference"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native UR10e 2F-140 gear-assembly slice preserves the analytic assembly workflow, "
        "but it does not include the upstream ROS inference transport or deployed-robot process stack."
    )


@configclass
class MacUR10eGearAssembly2F85RosInferenceEnvCfg(MacUR10eGearAssembly2F85EnvCfg):
    """Reduced mac-native 2F-85 gear-assembly configuration for the ROS-inference variant."""

    semantic_contract: str = "reduced-no-ros-inference"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native UR10e 2F-85 gear-assembly slice preserves the analytic assembly workflow, "
        "but it does not include the upstream ROS inference transport or deployed-robot process stack."
    )


@configclass
class MacOpenArmReachEnvCfg(MacFrankaReachEnvCfg):
    """Reduced mac-native OpenArm unimanual reach configuration."""

    action_scale: float = 0.32
    distance_reward_gain: float = 8.0
    default_joint_pos: tuple[float, ...] = (1.57, 0.0, -1.57, 1.57, 0.0, 0.0, 0.0)
    target_x_range: tuple[float, float] = (0.18, 0.42)
    target_y_range: tuple[float, float] = (-0.18, 0.18)
    target_z_range: tuple[float, float] = (0.10, 0.28)
    semantic_contract: str = "reduced-openarm-surrogate"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "This mac-native slice preserves the single-arm OpenArm reach workflow with a reduced "
        "analytic 7-DoF surrogate rather than the exact OpenArm morphology and upstream controller stack."
    )


@configclass
class MacOpenArmBiReachEnvCfg:
    """Reduced mac-native OpenArm bimanual reach configuration."""

    num_envs: int = 256
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 8.0
    action_space: int = 14
    observation_space: int = 46
    state_space: int = 0

    action_scale: float = 0.28
    action_rate_penalty_scale: float = -0.01
    joint_vel_penalty_scale: float = -0.0005
    reach_reward_scale: float = 1.15
    success_bonus: float = 3.5
    success_threshold: float = 0.08
    distance_reward_gain: float = 7.0

    joint_stiffness: float = 13.0
    joint_damping: float = 3.0
    joint_inertia: float = 1.15
    joint_reset_noise: float = 0.05
    default_joint_pos: tuple[float, ...] = (
        1.57,
        0.0,
        -1.57,
        1.57,
        0.0,
        0.0,
        0.0,
        -1.57,
        0.0,
        1.57,
        -1.57,
        0.0,
        0.0,
        0.0,
    )
    joint_lower_limits: tuple[float, ...] = (
        -2.6,
        -1.8,
        -2.6,
        -3.0,
        -2.6,
        -0.1,
        -2.6,
        -2.6,
        -1.8,
        -2.6,
        -3.0,
        -2.6,
        -0.1,
        -2.6,
    )
    joint_upper_limits: tuple[float, ...] = (
        2.6,
        1.8,
        2.6,
        -0.05,
        2.6,
        3.5,
        2.6,
        2.6,
        1.8,
        2.6,
        -0.05,
        2.6,
        3.5,
        2.6,
    )

    left_target_x_range: tuple[float, float] = (0.12, 0.36)
    left_target_y_range: tuple[float, float] = (0.08, 0.28)
    left_target_z_range: tuple[float, float] = (0.10, 0.28)
    right_target_x_range: tuple[float, float] = (0.12, 0.36)
    right_target_y_range: tuple[float, float] = (-0.28, -0.08)
    right_target_z_range: tuple[float, float] = (0.10, 0.28)
    left_arm_base_offset: tuple[float, float, float] = (0.0, 0.18, 0.0)
    right_arm_base_offset: tuple[float, float, float] = (0.0, -0.18, 0.0)

    semantic_contract: str = "reduced-openarm-bimanual-surrogate"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "This mac-native slice preserves the dual-arm OpenArm reach workflow with a reduced analytic "
        "two-arm surrogate rather than the exact bimanual controller and body-frame stack."
    )
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
class MacFrankaTeddyBearLiftEnvCfg(MacFrankaLiftEnvCfg):
    """Configuration for a reduced mac-native Franka teddy-bear lift slice."""

    cube_x_range: tuple[float, float] = (0.46, 0.62)
    cube_y_range: tuple[float, float] = (-0.10, 0.10)
    grasp_distance_threshold: float = 0.085
    grasp_offset_z: float = 0.07
    lift_success_height: float = 0.24
    reach_reward_scale: float = 1.4
    grasp_reward_scale: float = 0.65
    lift_reward_scale: float = 4.5
    lift_success_bonus: float = 5.0
    manipulated_object_label: str = "teddy-bear"


@configclass
class MacOpenArmLiftEnvCfg(MacFrankaLiftEnvCfg):
    """Reduced mac-native OpenArm cube-lift configuration."""

    action_scale: float = 0.24
    gripper_action_scale: float = 0.022
    default_joint_pos: tuple[float, ...] = (1.57, 0.0, -1.57, 1.57, 0.0, 0.0, 0.0, 0.022)
    joint_upper_limits: tuple[float, ...] = (2.6, 1.8, 2.6, -0.05, 2.6, 3.5, 2.6, 0.044)
    cube_x_range: tuple[float, float] = (0.20, 0.34)
    cube_y_range: tuple[float, float] = (-0.08, 0.08)
    grasp_distance_threshold: float = 0.06
    gripper_closed_threshold: float = 0.012
    grasp_offset_z: float = 0.045
    lift_success_height: float = 0.16
    semantic_contract: str = "reduced-openarm-surrogate"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "This mac-native slice preserves the OpenArm cube-lift workflow with reduced analytic "
        "grasp logic and a 7-DoF surrogate rather than the exact OpenArm kinematic chain."
    )


@configclass
class MacAgibotPlaceEnvCfg(MacOpenArmLiftEnvCfg):
    """Reduced mac-native Agibot place configuration built on the OpenArm-style surrogate."""

    observation_space: int = 34
    episode_length_s: float = 10.0

    place_target_x_range: tuple[float, float] = (0.18, 0.34)
    place_target_y_range: tuple[float, float] = (-0.12, 0.12)
    place_target_z_range: tuple[float, float] = (0.09, 0.20)
    place_xy_threshold: float = 0.045
    place_z_threshold: float = 0.035
    place_release_open_threshold: float = 0.018

    lift_reward_scale: float = 1.4
    place_align_reward_scale: float = 3.0
    place_distance_reward_gain: float = 8.5
    place_success_bonus: float = 6.0

    task_name: str = "agibot-place"
    semantic_contract: str = "reduced-agibot-place-surrogate"
    upstream_alias_semantics_preserved: bool = False
    manipulated_object_label: str = "place-object"
    target_label: str = "placement-target"
    contract_notes: str = (
        "This mac-native slice preserves the Agibot place workflow with reduced analytic grasp and "
        "placement logic instead of the exact RmpFlow-controlled Agibot scene."
    )


@configclass
class MacAgibotPlaceToy2BoxEnvCfg(MacAgibotPlaceEnvCfg):
    """Reduced mac-native Agibot toy-to-box place configuration."""

    task_name: str = "agibot-place-toy2box"
    cube_x_range: tuple[float, float] = (0.14, 0.28)
    cube_y_range: tuple[float, float] = (0.02, 0.16)
    place_target_x_range: tuple[float, float] = (0.24, 0.36)
    place_target_y_range: tuple[float, float] = (-0.02, 0.10)
    place_target_z_range: tuple[float, float] = (0.11, 0.17)
    manipulated_object_label: str = "toy"
    target_label: str = "box"
    contract_notes: str = (
        "This mac-native slice preserves the Agibot toy-to-box place workflow with reduced analytic "
        "grasp and placement logic rather than the exact Agibot arm, box geometry, and RmpFlow controller."
    )


@configclass
class MacAgibotPlaceUprightMugEnvCfg(MacAgibotPlaceEnvCfg):
    """Reduced mac-native Agibot upright-mug place configuration."""

    task_name: str = "agibot-place-upright-mug"
    cube_x_range: tuple[float, float] = (0.16, 0.30)
    cube_y_range: tuple[float, float] = (-0.16, -0.02)
    place_target_x_range: tuple[float, float] = (0.20, 0.32)
    place_target_y_range: tuple[float, float] = (-0.12, 0.00)
    place_target_z_range: tuple[float, float] = (0.10, 0.18)
    grasp_distance_threshold: float = 0.07
    grasp_offset_z: float = 0.055
    lift_success_height: float = 0.18
    place_xy_threshold: float = 0.04
    place_z_threshold: float = 0.03
    manipulated_object_label: str = "upright-mug"
    target_label: str = "mug-pad"
    contract_notes: str = (
        "This mac-native slice preserves the Agibot upright-mug place workflow with reduced analytic "
        "grasp and placement logic rather than the exact Agibot arm, mug pose-stability stack, and RmpFlow controller."
    )


@configclass
class MacFrankaStackEnvCfg(MacFrankaLiftEnvCfg):
    """Configuration for a reduced mac-native Franka cube-stacking slice."""

    observation_space: int = 33
    episode_length_s: float = 10.0
    task_name: str = "franka-stack"
    semantic_contract: str = "aligned"
    upstream_alias_semantics_preserved: bool = True
    contract_notes: str = "Analytic two-cube stack slice."

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
class MacFrankaStackInstanceRandomizeEnvCfg(MacFrankaStackEnvCfg):
    """Configuration for a reduced mac-native Franka instance-randomized stack slice."""

    observation_space: int = 35
    support_cube_x_range: tuple[float, float] = (0.44, 0.58)
    support_cube_y_range: tuple[float, float] = (-0.07, 0.07)
    movable_cube_offset_x_range: tuple[float, float] = (0.08, 0.18)
    movable_cube_offset_y_range: tuple[float, float] = (0.05, 0.14)
    task_name: str = "franka-stack-instance-randomize"
    variant_count: int = 4
    variant_labels: tuple[str, ...] = ("blue", "red", "yellow", "green")


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
class MacFrankaOpenDrawerEnvCfg(MacFrankaCabinetEnvCfg):
    """Configuration for a reduced mac-native Franka open-drawer slice."""


@configclass
class MacOpenArmOpenDrawerEnvCfg(MacFrankaOpenDrawerEnvCfg):
    """Reduced mac-native OpenArm open-drawer configuration."""

    action_scale: float = 0.24
    gripper_action_scale: float = 0.022
    default_joint_pos: tuple[float, ...] = (1.57, 0.0, -1.57, 1.57, 0.0, 0.0, 0.0, 0.022)
    joint_upper_limits: tuple[float, ...] = (2.6, 1.8, 2.6, -0.05, 2.6, 3.5, 2.6, 0.044)
    handle_anchor_x_range: tuple[float, float] = (0.22, 0.34)
    handle_anchor_y_range: tuple[float, float] = (-0.05, 0.05)
    handle_anchor_z_range: tuple[float, float] = (0.12, 0.18)
    handle_grasp_threshold: float = 0.055
    drawer_open_distance_max: float = 0.18
    drawer_success_distance: float = 0.14
    semantic_contract: str = "reduced-openarm-surrogate"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "This mac-native slice preserves the OpenArm drawer-opening workflow with reduced analytic "
        "grasp and drawer logic rather than the exact OpenArm cabinet scene and finger geometry."
    )


@configclass
class MacFrankaStackRgbEnvCfg(MacFrankaStackEnvCfg):
    """Configuration for a reduced three-cube Franka stack slice."""

    observation_space: int = 42
    episode_length_s: float = 12.0
    task_name: str = "franka-stack-rgb"
    semantic_contract: str = "aligned"
    upstream_alias_semantics_preserved: bool = True
    contract_notes: str = "Analytic three-cube sequential stack slice."

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


@configclass
class MacFrankaBinStackEnvCfg(MacFrankaStackRgbEnvCfg):
    """Configuration for a reduced bin-anchored three-cube Franka stack slice.

    The upstream task id includes ``Mimic`` semantics. The mac-native slice does not
    implement imitation or demonstration-conditioned behavior, so this config carries
    an explicit reduced-contract marker.
    """

    observation_space: int = 45
    task_name: str = "franka-bin-stack"
    semantic_contract: str = "reduced-no-mimic"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "Upstream mimic/imitation semantics are not implemented on mac-sim; "
        "this task resolves to the reduced bin-anchored stack slice."
    )
    bin_anchor_observation_mode: str = "mirrored-support-anchor-tail"
    cube_x_range: tuple[float, float] = (0.36, 0.74)
    cube_y_range: tuple[float, float] = (-0.22, 0.22)
    bin_anchor_x_range: tuple[float, float] = (0.38, 0.44)
    bin_anchor_y_range: tuple[float, float] = (-0.04, 0.04)
    middle_cube_x_range: tuple[float, float] = (0.62, 0.70)
    middle_cube_y_abs_range: tuple[float, float] = (0.10, 0.18)
    top_cube_x_range: tuple[float, float] = (0.62, 0.70)
    top_cube_y_abs_range: tuple[float, float] = (0.10, 0.18)


@configclass
class MacFrankaBinStackPickPlaceEnvCfg(MacFrankaBinStackEnvCfg):
    """Configuration for the reduced pick-place surrogate mapped onto the bin-anchored stack substrate."""

    task_name: str = "franka-pick-place-surrogate"
    semantic_contract: str = "reduced-pick-place-surrogate"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native pick-place surrogate resolves to the reduced bin-anchored stack substrate "
        "instead of the upstream pick/place scene, so the public task stays available without "
        "pretending exact object identity or pick/place fixture parity."
    )


@configclass
class MacUR10LongSuctionStackEnvCfg(MacUR10ReachEnvCfg):
    """Reduced mac-native UR10 long-suction three-cube stack configuration."""

    action_space: int = 7
    observation_space: int = 40
    episode_length_s: float = 12.0
    action_scale: float = 0.07
    default_joint_pos: tuple[float, ...] = (0.0, -1.35, 1.55, -1.75, -1.57, 0.0)
    joint_lower_limits: tuple[float, ...] = (
        -2.0 * math.pi,
        -math.pi,
        -math.pi,
        -2.0 * math.pi,
        -2.0 * math.pi,
        -2.0 * math.pi,
    )
    joint_upper_limits: tuple[float, ...] = (
        2.0 * math.pi,
        0.0,
        math.pi,
        2.0 * math.pi,
        2.0 * math.pi,
        2.0 * math.pi,
    )
    task_name: str = "ur10-long-suction-stack"
    semantic_contract: str = "reduced-analytic-suction-stack"
    upstream_alias_semantics_preserved: bool = False
    suction_variant: str = "long"
    contract_notes: str = (
        "This mac-native slice preserves the UR10 long-suction three-cube stack workflow with "
        "analytic pose tracking and suction-state surrogates instead of the upstream CPU-only "
        "surface-gripper simulation and full UR10 suction stack."
    )

    support_cube_x_range: tuple[float, float] = (0.42, 0.54)
    support_cube_y_range: tuple[float, float] = (-0.06, 0.06)
    middle_cube_offset_x_range: tuple[float, float] = (0.10, 0.16)
    middle_cube_offset_y_range: tuple[float, float] = (0.08, 0.14)
    top_cube_offset_x_range: tuple[float, float] = (-0.16, -0.10)
    top_cube_offset_y_range: tuple[float, float] = (0.08, 0.14)

    gripper_lower_limit: float = -1.0
    gripper_upper_limit: float = 1.0
    default_gripper_state: float = 1.0
    gripper_closed_threshold: float = -0.5
    stack_release_open_threshold: float = 0.5
    grasp_distance_threshold: float = 0.11
    grasp_offset_z: float = 0.08
    stack_offset_z: float = 0.04
    stack_xy_threshold: float = 0.045
    stack_z_threshold: float = 0.035
    table_height: float = 0.0203

    grasp_reward_scale: float = 0.4
    lift_reward_scale: float = 1.1
    middle_stage_bonus: float = 2.0
    top_stack_align_reward_scale: float = 2.8
    top_stack_distance_reward_gain: float = 9.5
    stack_success_bonus: float = 8.0


@configclass
class MacUR10ShortSuctionStackEnvCfg(MacUR10LongSuctionStackEnvCfg):
    """Reduced mac-native UR10 short-suction three-cube stack configuration."""

    task_name: str = "ur10-short-suction-stack"
    suction_variant: str = "short"
    contract_notes: str = (
        "This mac-native slice preserves the UR10 short-suction three-cube stack workflow with "
        "analytic pose tracking and suction-state surrogates instead of the upstream CPU-only "
        "surface-gripper simulation and full UR10 suction stack."
    )
    support_cube_x_range: tuple[float, float] = (0.40, 0.52)
    middle_cube_offset_x_range: tuple[float, float] = (0.09, 0.15)
    top_cube_offset_x_range: tuple[float, float] = (-0.15, -0.09)
    grasp_distance_threshold: float = 0.095
    grasp_offset_z: float = 0.06
    stack_xy_threshold: float = 0.04
    top_stack_align_reward_scale: float = 3.0


@configclass
class MacFrankaStackBlueprintEnvCfg(MacFrankaStackEnvCfg):
    """Configuration for the reduced Franka stack blueprint variant."""

    task_name: str = "franka-stack-blueprint"
    semantic_contract: str = "reduced-no-blueprint"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native blueprint variant preserves the Franka stack workflow but does not include "
        "the upstream blueprint-conditioned generation semantics."
    )


@configclass
class MacFrankaStackSkillgenEnvCfg(MacFrankaStackEnvCfg):
    """Configuration for the reduced Franka stack skillgen variant."""

    task_name: str = "franka-stack-skillgen"
    semantic_contract: str = "reduced-no-skillgen"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native skillgen variant preserves the Franka stack workflow but does not include "
        "the upstream skill-generation or demonstration-conditioned behavior."
    )


@configclass
class MacFrankaStackVisuomotorEnvCfg(MacFrankaStackRgbEnvCfg):
    """Configuration for the reduced Franka stack visuomotor variant."""

    task_name: str = "franka-stack-visuomotor"
    semantic_contract: str = "reduced-visuomotor-surrogate"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native visuomotor variant preserves the three-cube stack workflow with synthetic RGB "
        "observations and analytic object dynamics instead of the upstream robomimic image stack."
    )


@configclass
class MacFrankaStackVisuomotorCosmosEnvCfg(MacFrankaStackRgbEnvCfg):
    """Configuration for the reduced Franka stack visuomotor-cosmos variant."""

    task_name: str = "franka-stack-visuomotor-cosmos"
    semantic_contract: str = "reduced-no-cosmos"
    upstream_alias_semantics_preserved: bool = False
    contract_notes: str = (
        "The mac-native visuomotor-cosmos variant preserves the three-cube stack workflow with synthetic RGB "
        "observations, but it does not include the upstream robomimic visuomotor stack or the Cosmos multimodal "
        "image contract for RGB, segmentation, normals, and depth channels."
    )
