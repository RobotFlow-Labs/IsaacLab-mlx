# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime backend selection, capability metadata, and narrow device adapters."""

from __future__ import annotations

import builtins
import importlib.util
import os
import pickle
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

ComputeBackendName = Literal["torch-cuda", "mlx"]
SimBackendName = Literal["isaacsim", "mac-sim"]
KernelBackendName = Literal["warp", "metal", "cpu"]
SensorBackendName = Literal["isaacsim-sensors", "mac-sensors", "cpu"]
PlannerBackendName = Literal["isaacsim-planners", "mac-planners", "none"]

ENV_COMPUTE_BACKEND = "ISAACLAB_COMPUTE_BACKEND"
ENV_SIM_BACKEND = "ISAACLAB_SIM_BACKEND"
ENV_KERNEL_BACKEND = "ISAACLAB_KERNEL_BACKEND"


class UnsupportedBackendError(RuntimeError):
    """Raised when the selected runtime backends are not supported by the current code path."""


class UnsupportedRuntimeFeatureError(UnsupportedBackendError):
    """Raised when a backend pair exists but lacks a requested capability."""


@dataclass(frozen=True)
class ComputeCapabilities:
    """Capabilities exposed by a compute backend."""

    autograd: bool = True
    checkpoint_io: bool = True
    custom_kernels: bool = False
    torch_interop: bool = False


@dataclass(frozen=True)
class SimCapabilities:
    """Capabilities exposed by a simulation backend."""

    batched_stepping: bool = False
    articulated_rigid_bodies: bool = False
    contacts: bool = False
    proprioceptive_observations: bool = False
    cameras: bool = False
    planners: bool = False


@dataclass(frozen=True)
class KernelCapabilities:
    """Capabilities exposed by a kernel backend."""

    custom_kernels: bool = False
    raycast: bool = False
    fabric_transforms: bool = False
    mesh_queries: bool = False
    cpu_fallback: bool = False


@dataclass(frozen=True)
class SensorCapabilities:
    """Capabilities exposed by a sensor backend."""

    proprioception: bool = True
    raycast: bool = False
    cameras: bool = False
    depth: bool = False
    segmentation: bool = False
    rgb: bool = False


@dataclass(frozen=True)
class PlannerCapabilities:
    """Capabilities exposed by a planner backend."""

    motion_generation: bool = False
    inverse_kinematics: bool = False
    batched_planning: bool = False


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Capability set for the selected compute/sim pair."""

    compute: ComputeCapabilities
    sim: SimCapabilities
    kernel: KernelCapabilities
    sensor: SensorCapabilities
    planner: PlannerCapabilities


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved runtime backend selection."""

    compute_backend: ComputeBackendName
    sim_backend: SimBackendName
    kernel_backend: KernelBackendName
    device: str


@dataclass(frozen=True)
class ArticulationCapabilities:
    """Operations the simulator exposes for articulated assets."""

    joint_state_io: bool = False
    root_state_io: bool = False
    effort_targets: bool = False
    batched_views: bool = False


@dataclass(frozen=True)
class SimBackendContract:
    """Reference contract for the simulator APIs IsaacLab expects to call."""

    reset_signature: str
    step_signature: str
    articulations: ArticulationCapabilities


class ComputeBackend(ABC):
    """Abstract compute backend seam for torch/MLX implementations."""

    name: ComputeBackendName
    capabilities: ComputeCapabilities

    @abstractmethod
    def configure_device(self, device: str) -> None:
        """Prepare the backend for the selected device."""

    @abstractmethod
    def seed(self, seed: int) -> int:
        """Seed the backend and return the effective seed."""

    @abstractmethod
    def save_checkpoint(self, path: str, payload: Any) -> None:
        """Persist a backend-native checkpoint."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> Any:
        """Load a backend-native checkpoint."""


class TorchCudaComputeBackend(ComputeBackend):
    """Torch/CUDA compute adapter for the upstream runtime."""

    name: ComputeBackendName = "torch-cuda"
    capabilities = ComputeCapabilities(
        autograd=True,
        checkpoint_io=True,
        custom_kernels=True,
        torch_interop=True,
    )

    def configure_device(self, device: str) -> None:
        if not device.startswith("cuda"):
            return
        import torch

        torch.cuda.set_device(device)

    def seed(self, seed: int) -> int:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
        return seed

    def save_checkpoint(self, path: str, payload: Any) -> None:
        import torch

        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> Any:
        import torch

        return torch.load(path, map_location="cpu")


class MlxComputeBackend(ComputeBackend):
    """MLX compute adapter for Apple Silicon runtime."""

    name: ComputeBackendName = "mlx"
    capabilities = ComputeCapabilities(
        autograd=True,
        checkpoint_io=True,
        custom_kernels=True,
        torch_interop=False,
    )

    def configure_device(self, device: str) -> None:
        # MLX manages device placement implicitly through the runtime.
        del device

    def seed(self, seed: int) -> int:
        if not is_mlx_available():
            raise UnsupportedRuntimeFeatureError("MLX backend selected but `mlx` is not importable.")
        import mlx.core as mx

        mx.random.seed(seed)
        return seed

    def save_checkpoint(self, path: str, payload: Any) -> None:
        # Use pickle as a backend-agnostic transport for now. Callers control payload structure.
        with open(path, "wb") as file:
            pickle.dump(payload, file)

    def load_checkpoint(self, path: str) -> Any:
        with open(path, "rb") as file:
            return pickle.load(file)


class KernelBackend(ABC):
    """Abstract backend seam for Warp, Metal, and CPU kernel implementations."""

    name: KernelBackendName
    capabilities: KernelCapabilities

    def require_feature(self, feature: str) -> None:
        if getattr(self.capabilities, feature, False):
            return
        raise UnsupportedRuntimeFeatureError(
            f"Kernel backend '{self.name}' does not implement required feature '{feature}'."
        )

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable backend state."""


class WarpKernelBackend(KernelBackend):
    """Kernel backend for the upstream Warp/CUDA path."""

    name: KernelBackendName = "warp"
    capabilities = KernelCapabilities(
        custom_kernels=True,
        raycast=True,
        fabric_transforms=True,
        mesh_queries=True,
        cpu_fallback=False,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class MetalKernelBackend(KernelBackend):
    """Kernel backend for MLX plus custom Metal kernels on Apple Silicon."""

    name: KernelBackendName = "metal"
    capabilities = KernelCapabilities(
        custom_kernels=True,
        raycast=True,
        fabric_transforms=False,
        mesh_queries=True,
        cpu_fallback=True,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class CpuKernelBackend(KernelBackend):
    """Fallback kernel backend used during correctness bring-up."""

    name: KernelBackendName = "cpu"
    capabilities = KernelCapabilities(
        custom_kernels=False,
        raycast=False,
        fabric_transforms=False,
        mesh_queries=False,
        cpu_fallback=True,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class SimBackend(ABC):
    """Abstract simulation backend seam for Isaac Sim and future macOS simulators."""

    name: SimBackendName
    capabilities: SimCapabilities
    contract: SimBackendContract

    @abstractmethod
    def reset(self, *, soft: bool = False) -> Any:
        """Reset the active simulation."""

    @abstractmethod
    def step(self, *, render: bool = True, update_fabric: bool = False) -> Any:
        """Step the active simulation."""

    @abstractmethod
    def get_joint_state(self, articulation: Any) -> tuple[Any, Any]:
        """Read batched joint position and velocity state from an articulation."""

    @abstractmethod
    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Sequence[int] | None = None,
    ) -> None:
        """Write joint effort targets to the articulation."""

    @abstractmethod
    def write_joint_state(
        self,
        articulation: Any,
        joint_pos: Any,
        joint_vel: Any,
        *,
        joint_acc: Any | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write joint state back into the simulator."""

    @abstractmethod
    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Sequence[int] | None = None) -> None:
        """Write root pose back into the simulator."""

    @abstractmethod
    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root velocity back into the simulator."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable simulator state."""


class IsaacSimBackend(SimBackend):
    """Adapter around the upstream Isaac Sim simulation/articulation APIs."""

    name: SimBackendName = "isaacsim"
    capabilities = SimCapabilities(
        batched_stepping=True,
        articulated_rigid_bodies=True,
        contacts=True,
        proprioceptive_observations=True,
        cameras=True,
        planners=True,
    )
    contract = SimBackendContract(
        reset_signature="reset(soft: bool = False) -> None",
        step_signature="step(render: bool = True, update_fabric: bool = False) -> None",
        articulations=ArticulationCapabilities(
            joint_state_io=True,
            root_state_io=True,
            effort_targets=True,
            batched_views=True,
        ),
    )

    def __init__(self, simulation_context: Any | None = None):
        self._simulation_context = simulation_context

    def attach(self, simulation_context: Any) -> None:
        """Attach a concrete Isaac Sim simulation context."""
        self._simulation_context = simulation_context

    def _require_context(self) -> Any:
        if self._simulation_context is None:
            raise UnsupportedRuntimeFeatureError(
                "IsaacSim backend requires an attached simulation context before calling reset/step."
            )
        return self._simulation_context

    def reset(self, *, soft: bool = False) -> Any:
        return self._require_context().reset(soft=soft)

    def step(self, *, render: bool = True, update_fabric: bool = False) -> Any:
        return self._require_context().step(render=render, update_fabric=update_fabric)

    def get_joint_state(self, articulation: Any) -> tuple[Any, Any]:
        return articulation.data.joint_pos, articulation.data.joint_vel

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Sequence[int] | None = None,
    ) -> None:
        articulation.set_joint_effort_target(efforts, joint_ids=joint_ids)

    def write_joint_state(
        self,
        articulation: Any,
        joint_pos: Any,
        joint_vel: Any,
        *,
        joint_acc: Any | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        articulation.write_joint_state_to_sim(joint_pos, joint_vel, joint_acc, env_ids)

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Sequence[int] | None = None) -> None:
        articulation.write_root_pose_to_sim(root_pose, env_ids)

    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        articulation.write_root_velocity_to_sim(root_velocity, env_ids)

    def state_dict(self) -> dict[str, Any]:
        simulation_context = self._simulation_context
        if simulation_context is None:
            return {"attached": False, "backend": self.name}
        return {
            "attached": True,
            "backend": self.name,
            "device": getattr(simulation_context, "device", None),
            "render_mode": getattr(getattr(simulation_context, "render_mode", None), "name", None),
        }


class MacSimBackend(SimBackend):
    """Placeholder adapter for the future macOS-native simulator."""

    name: SimBackendName = "mac-sim"
    capabilities = SimCapabilities()
    contract = SimBackendContract(
        reset_signature="reset(soft: bool = False) -> Any",
        step_signature="step(render: bool = True, update_fabric: bool = False) -> Any",
        articulations=ArticulationCapabilities(),
    )

    def _unimplemented(self, operation: str) -> None:
        raise UnsupportedRuntimeFeatureError(
            f"`mac-sim` does not implement `{operation}` yet. The public port foundation is in place,"
            " but the macOS simulator adapter still needs a concrete implementation."
        )

    def reset(self, *, soft: bool = False) -> Any:
        self._unimplemented("reset")

    def step(self, *, render: bool = True, update_fabric: bool = False) -> Any:
        self._unimplemented("step")

    def get_joint_state(self, articulation: Any) -> tuple[Any, Any]:
        self._unimplemented("get_joint_state")

    def set_joint_effort_target(
        self,
        articulation: Any,
        efforts: Any,
        *,
        joint_ids: Sequence[int] | None = None,
    ) -> None:
        self._unimplemented("set_joint_effort_target")

    def write_joint_state(
        self,
        articulation: Any,
        joint_pos: Any,
        joint_vel: Any,
        *,
        joint_acc: Any | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        self._unimplemented("write_joint_state")

    def write_root_pose(self, articulation: Any, root_pose: Any, *, env_ids: Sequence[int] | None = None) -> None:
        self._unimplemented("write_root_pose")

    def write_root_velocity(
        self,
        articulation: Any,
        root_velocity: Any,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        self._unimplemented("write_root_velocity")

    def state_dict(self) -> dict[str, Any]:
        return {"attached": False, "backend": self.name}


class SensorBackend(ABC):
    """Abstract backend seam for sensor implementations."""

    name: SensorBackendName
    capabilities: SensorCapabilities

    def require_feature(self, feature: str) -> None:
        if getattr(self.capabilities, feature, False):
            return
        raise UnsupportedRuntimeFeatureError(
            f"Sensor backend '{self.name}' does not implement required feature '{feature}'."
        )

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable backend state."""


class IsaacSimSensorBackend(SensorBackend):
    """Sensor backend for the upstream Isaac Sim runtime."""

    name: SensorBackendName = "isaacsim-sensors"
    capabilities = SensorCapabilities(
        proprioception=True,
        raycast=True,
        cameras=True,
        depth=True,
        segmentation=True,
        rgb=True,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class MacSensorBackend(SensorBackend):
    """Sensor backend for the mac-native simulator path."""

    name: SensorBackendName = "mac-sensors"
    capabilities = SensorCapabilities(
        proprioception=True,
        raycast=True,
        cameras=True,
        depth=True,
        segmentation=False,
        rgb=True,
    )

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "capabilities": self.capabilities.__dict__,
            "implementation": "analytic-plane-raycast+external-stereo-depth",
        }


class CpuSensorBackend(SensorBackend):
    """Fallback sensor backend for CPU bring-up paths."""

    name: SensorBackendName = "cpu"
    capabilities = SensorCapabilities(
        proprioception=True,
        raycast=False,
        cameras=False,
        depth=False,
        segmentation=False,
        rgb=False,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class PlannerBackend(ABC):
    """Abstract backend seam for planners and motion generation."""

    name: PlannerBackendName
    capabilities: PlannerCapabilities

    def require_feature(self, feature: str) -> None:
        if getattr(self.capabilities, feature, False):
            return
        raise UnsupportedRuntimeFeatureError(
            f"Planner backend '{self.name}' does not implement required feature '{feature}'."
        )

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable backend state."""


class IsaacSimPlannerBackend(PlannerBackend):
    """Planner backend for the upstream Isaac Sim path."""

    name: PlannerBackendName = "isaacsim-planners"
    capabilities = PlannerCapabilities(
        motion_generation=True,
        inverse_kinematics=True,
        batched_planning=True,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class MacPlannerBackend(PlannerBackend):
    """Placeholder planner backend for the mac-native path."""

    name: PlannerBackendName = "mac-planners"
    capabilities = PlannerCapabilities(
        motion_generation=False,
        inverse_kinematics=False,
        batched_planning=False,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


class NullPlannerBackend(PlannerBackend):
    """Planner backend used when no planner path exists yet."""

    name: PlannerBackendName = "none"
    capabilities = PlannerCapabilities(
        motion_generation=False,
        inverse_kinematics=False,
        batched_planning=False,
    )

    def state_dict(self) -> dict[str, Any]:
        return {"backend": self.name, "capabilities": self.capabilities.__dict__}


def is_mlx_available() -> bool:
    """Return True when MLX can be imported in the current interpreter."""
    return importlib.util.find_spec("mlx") is not None


def is_isaacsim_available() -> bool:
    """Return True when Isaac Sim Python modules can be imported."""
    return importlib.util.find_spec("isaacsim") is not None


def is_apple_silicon() -> bool:
    """Return True on Apple Silicon macOS hosts."""
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def normalize_compute_backend(name: str | None) -> ComputeBackendName | None:
    """Normalize public compute backend aliases."""
    if name is None:
        return None
    normalized = name.strip().lower()
    if normalized in {"torch", "torch-cuda", "cuda"}:
        return "torch-cuda"
    if normalized == "mlx":
        return "mlx"
    raise ValueError(f"Unsupported compute backend: {name!r}. Expected one of: 'torch-cuda', 'mlx'.")


def normalize_sim_backend(name: str | None) -> SimBackendName | None:
    """Normalize public simulation backend aliases."""
    if name is None:
        return None
    normalized = name.strip().lower()
    if normalized in {"isaacsim", "isaac-sim"}:
        return "isaacsim"
    if normalized in {"mac-sim", "macsim", "mac"}:
        return "mac-sim"
    raise ValueError(f"Unsupported sim backend: {name!r}. Expected one of: 'isaacsim', 'mac-sim'.")


def normalize_kernel_backend(name: str | None) -> KernelBackendName | None:
    """Normalize public kernel backend aliases."""
    if name is None:
        return None
    normalized = name.strip().lower()
    if normalized == "warp":
        return "warp"
    if normalized in {"metal", "mlx-metal"}:
        return "metal"
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported kernel backend: {name!r}. Expected one of: 'warp', 'metal', 'cpu'.")


def _default_compute_backend(device: str, sim_backend: SimBackendName) -> ComputeBackendName:
    if sim_backend == "mac-sim":
        if is_apple_silicon() or is_mlx_available():
            return "mlx"
    if device.startswith("cuda"):
        return "torch-cuda"
    return "torch-cuda"


def _default_kernel_backend(sim_backend: SimBackendName) -> KernelBackendName:
    if sim_backend == "mac-sim":
        if is_apple_silicon() or is_mlx_available():
            return "metal"
        return "cpu"
    return "warp"


def _sensor_backend_name(runtime: RuntimeSelection) -> SensorBackendName:
    if runtime.sim_backend == "isaacsim":
        return "isaacsim-sensors"
    if runtime.kernel_backend == "cpu":
        return "cpu"
    return "mac-sensors"


def _planner_backend_name(runtime: RuntimeSelection) -> PlannerBackendName:
    if runtime.sim_backend == "isaacsim":
        return "isaacsim-planners"
    if runtime.compute_backend == "mlx":
        return "mac-planners"
    return "none"


def resolve_runtime_selection(
    compute_backend: str | None = None,
    sim_backend: str | None = None,
    device: str | None = None,
    kernel_backend: str | None = None,
) -> RuntimeSelection:
    """Resolve compute/sim backends from explicit inputs plus environment defaults."""
    normalized_sim = normalize_sim_backend(sim_backend or os.environ.get(ENV_SIM_BACKEND)) or "isaacsim"
    device = device or ("cpu" if normalized_sim == "mac-sim" else "cuda:0")
    normalized_compute = (
        normalize_compute_backend(compute_backend or os.environ.get(ENV_COMPUTE_BACKEND))
        or _default_compute_backend(device, normalized_sim)
    )
    normalized_kernel = (
        normalize_kernel_backend(kernel_backend or os.environ.get(ENV_KERNEL_BACKEND))
        or _default_kernel_backend(normalized_sim)
    )
    return RuntimeSelection(
        compute_backend=normalized_compute,
        sim_backend=normalized_sim,
        kernel_backend=normalized_kernel,
        device=device,
    )


def set_runtime_selection(runtime: RuntimeSelection) -> RuntimeSelection:
    """Persist the runtime selection in environment variables and builtins."""
    os.environ[ENV_COMPUTE_BACKEND] = runtime.compute_backend
    os.environ[ENV_SIM_BACKEND] = runtime.sim_backend
    os.environ[ENV_KERNEL_BACKEND] = runtime.kernel_backend
    builtins.ISAACLAB_COMPUTE_BACKEND = runtime.compute_backend
    builtins.ISAACLAB_SIM_BACKEND = runtime.sim_backend
    builtins.ISAACLAB_KERNEL_BACKEND = runtime.kernel_backend
    builtins.ISAACLAB_RUNTIME_DEVICE = runtime.device
    return runtime


def current_runtime(default_device: str | None = None) -> RuntimeSelection:
    """Return the currently selected runtime from process state."""
    return resolve_runtime_selection(device=getattr(builtins, "ISAACLAB_RUNTIME_DEVICE", default_device))


def current_runtime_capabilities(runtime: RuntimeSelection | None = None) -> RuntimeCapabilities:
    """Return the capability set for the selected runtime."""
    runtime = runtime or current_runtime()
    if runtime.compute_backend == "mlx":
        compute = ComputeCapabilities(
            autograd=is_mlx_available(),
            checkpoint_io=is_mlx_available(),
            custom_kernels=is_mlx_available(),
            torch_interop=False,
        )
    else:
        compute = ComputeCapabilities(
            autograd=True,
            checkpoint_io=True,
            custom_kernels=True,
            torch_interop=True,
        )

    if runtime.sim_backend == "isaacsim":
        sim = SimCapabilities(
            batched_stepping=True,
            articulated_rigid_bodies=True,
            contacts=True,
            proprioceptive_observations=True,
            cameras=True,
            planners=True,
        )
    else:
        sim = SimCapabilities()

    if runtime.kernel_backend == "warp":
        kernel = WarpKernelBackend.capabilities
    elif runtime.kernel_backend == "metal":
        kernel = MetalKernelBackend.capabilities
    else:
        kernel = CpuKernelBackend.capabilities

    sensor_name = _sensor_backend_name(runtime)
    if sensor_name == "isaacsim-sensors":
        sensor = IsaacSimSensorBackend.capabilities
    elif sensor_name == "mac-sensors":
        sensor = MacSensorBackend.capabilities
    else:
        sensor = CpuSensorBackend.capabilities

    planner_name = _planner_backend_name(runtime)
    if planner_name == "isaacsim-planners":
        planner = IsaacSimPlannerBackend.capabilities
    elif planner_name == "mac-planners":
        planner = MacPlannerBackend.capabilities
    else:
        planner = NullPlannerBackend.capabilities

    return RuntimeCapabilities(
        compute=compute,
        sim=sim,
        kernel=kernel,
        sensor=sensor,
        planner=planner,
    )


def create_sim_backend(runtime: RuntimeSelection | None = None, *, simulation_context: Any | None = None) -> SimBackend:
    """Instantiate the simulator adapter for the active runtime."""
    runtime = runtime or current_runtime()
    if runtime.sim_backend == "isaacsim":
        return IsaacSimBackend(simulation_context=simulation_context)
    return MacSimBackend()


def create_kernel_backend(runtime: RuntimeSelection | None = None) -> KernelBackend:
    """Instantiate the kernel adapter for the active runtime."""
    runtime = runtime or current_runtime()
    if runtime.kernel_backend == "warp":
        return WarpKernelBackend()
    if runtime.kernel_backend == "metal":
        return MetalKernelBackend()
    return CpuKernelBackend()


def create_sensor_backend(runtime: RuntimeSelection | None = None) -> SensorBackend:
    """Instantiate the sensor adapter for the active runtime."""
    runtime = runtime or current_runtime()
    sensor_backend = _sensor_backend_name(runtime)
    if sensor_backend == "isaacsim-sensors":
        return IsaacSimSensorBackend()
    if sensor_backend == "mac-sensors":
        return MacSensorBackend()
    return CpuSensorBackend()


def create_planner_backend(runtime: RuntimeSelection | None = None) -> PlannerBackend:
    """Instantiate the planner adapter for the active runtime."""
    runtime = runtime or current_runtime()
    planner_backend = _planner_backend_name(runtime)
    if planner_backend == "isaacsim-planners":
        return IsaacSimPlannerBackend()
    if planner_backend == "mac-planners":
        return MacPlannerBackend()
    return NullPlannerBackend()


def create_compute_backend(runtime: RuntimeSelection | None = None) -> ComputeBackend:
    """Instantiate the compute adapter for the active runtime."""
    runtime = runtime or current_runtime()
    if runtime.compute_backend == "torch-cuda":
        return TorchCudaComputeBackend()
    return MlxComputeBackend()


def get_runtime_state(runtime: RuntimeSelection | None = None) -> dict[str, Any]:
    """Return runtime selection plus capability metadata for diagnostics and tests."""
    runtime = runtime or current_runtime()
    capabilities = current_runtime_capabilities(runtime)
    return {
        "compute_backend": runtime.compute_backend,
        "sim_backend": runtime.sim_backend,
        "kernel_backend": runtime.kernel_backend,
        "sensor_backend": _sensor_backend_name(runtime),
        "planner_backend": _planner_backend_name(runtime),
        "device": runtime.device,
        "capabilities": {
            "compute": capabilities.compute.__dict__,
            "sim": capabilities.sim.__dict__,
            "kernel": capabilities.kernel.__dict__,
            "sensor": capabilities.sensor.__dict__,
            "planner": capabilities.planner.__dict__,
        },
    }


def require_runtime_backends(
    caller: str,
    *,
    compute_backend: ComputeBackendName | None = None,
    sim_backend: SimBackendName | None = None,
    kernel_backend: KernelBackendName | None = None,
) -> RuntimeSelection:
    """Validate the active runtime against required backends."""
    runtime = current_runtime()
    if compute_backend is not None and runtime.compute_backend != compute_backend:
        raise UnsupportedBackendError(
            f"{caller} requires compute backend '{compute_backend}', but the active backend is"
            f" '{runtime.compute_backend}'."
        )
    if sim_backend is not None and runtime.sim_backend != sim_backend:
        raise UnsupportedBackendError(
            f"{caller} requires sim backend '{sim_backend}', but the active backend is '{runtime.sim_backend}'."
        )
    if kernel_backend is not None and runtime.kernel_backend != kernel_backend:
        raise UnsupportedBackendError(
            f"{caller} requires kernel backend '{kernel_backend}', but the active backend is"
            f" '{runtime.kernel_backend}'."
        )
    return runtime


def configure_torch_device(device: str, runtime: RuntimeSelection | None = None) -> None:
    """Route torch CUDA device selection through the active compute backend."""
    runtime = runtime or current_runtime(default_device=device)
    if runtime.compute_backend != "torch-cuda":
        return
    if not device.startswith("cuda"):
        return

    import torch

    torch.cuda.set_device(device)
