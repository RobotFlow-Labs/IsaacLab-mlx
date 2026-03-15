# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime backend selection and capability helpers."""

from .benchmark_reporting import (
    build_benchmark_dashboard,
    build_benchmark_trend,
    build_semantic_drift_snapshot,
    compare_semantic_drift,
)
from .kernel_compat import (
    add_forces_and_torques_at_position_mlx,
    detect_cpu_fallback,
    reshape_tiled_image_mlx,
    set_forces_and_torques_at_position_mlx,
)
from .runtime import (
    ArticulationCapabilities,
    ENV_COMPUTE_BACKEND,
    ENV_KERNEL_BACKEND,
    ENV_SIM_BACKEND,
    ComputeBackend,
    ComputeCapabilities,
    CpuKernelBackend,
    IsaacSimBackend,
    IsaacSimPlannerBackend,
    IsaacSimSensorBackend,
    KernelBackend,
    KernelCapabilities,
    MacPlannerBackend,
    MacSensorBackend,
    MlxComputeBackend,
    MacSimBackend,
    MetalKernelBackend,
    PlannerBackend,
    PlannerCapabilities,
    RuntimeCapabilities,
    RuntimeSelection,
    SensorBackend,
    SensorCapabilities,
    SimBackend,
    SimBackendContract,
    SimCapabilities,
    TorchCudaComputeBackend,
    UnsupportedBackendError,
    UnsupportedRuntimeFeatureError,
    WarpKernelBackend,
    configure_torch_device,
    create_compute_backend,
    create_kernel_backend,
    create_planner_backend,
    create_sensor_backend,
    create_sim_backend,
    build_runtime_diagnostics_payload,
    current_runtime,
    current_runtime_capabilities,
    get_runtime_state,
    is_apple_silicon,
    is_isaacsim_available,
    is_mlx_available,
    normalize_kernel_backend,
    resolve_runtime_selection,
    set_runtime_selection,
)
from .supported_tasks import (
    MacNativeTaskSpec,
    benchmark_task_groups,
    list_current_mac_native_tasks,
    list_public_mlx_tasks,
    list_trainable_mlx_tasks,
    mac_native_task_spec_map,
    mac_native_task_specs,
    supported_task_surface_summary,
)
from .planner_compat import (
    JointMotionPlan,
    JointMotionPlanRequest,
    PlannerWorldObstacle,
    PlannerWorldState,
    interpolate_joint_motion,
    interpolate_joint_motion_batch,
)
from .ros2_compat import (
    Ros2JsonlBridge,
    Ros2MessageEnvelope,
    Ros2ProcessBridge,
    joint_motion_plan_batch_from_ros_envelopes,
    joint_motion_plan_batch_to_ros_envelopes,
    joint_motion_plan_from_ros_envelope,
    joint_motion_plan_to_ros_envelope,
    planner_world_state_batch_from_ros_envelopes,
    planner_world_state_batch_to_ros_envelopes,
    planner_world_state_from_ros_envelope,
    planner_world_state_to_ros_envelope,
    ros2_cli_available,
)
