# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime backend selection, capability metadata, and narrow device adapters."""

from __future__ import annotations

import builtins
import importlib.util
import os
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

ComputeBackendName = Literal["torch-cuda", "mlx"]
SimBackendName = Literal["isaacsim", "mac-sim"]

ENV_COMPUTE_BACKEND = "ISAACLAB_COMPUTE_BACKEND"
ENV_SIM_BACKEND = "ISAACLAB_SIM_BACKEND"


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
class RuntimeCapabilities:
    """Capability set for the selected compute/sim pair."""

    compute: ComputeCapabilities
    sim: SimCapabilities


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved runtime backend selection."""

    compute_backend: ComputeBackendName
    sim_backend: SimBackendName
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


def _default_compute_backend(device: str, sim_backend: SimBackendName) -> ComputeBackendName:
    if sim_backend == "mac-sim":
        if is_apple_silicon() or is_mlx_available():
            return "mlx"
    if device.startswith("cuda"):
        return "torch-cuda"
    return "torch-cuda"


def resolve_runtime_selection(
    compute_backend: str | None = None,
    sim_backend: str | None = None,
    device: str | None = None,
) -> RuntimeSelection:
    """Resolve compute/sim backends from explicit inputs plus environment defaults."""
    device = device or "cuda:0"
    normalized_sim = normalize_sim_backend(sim_backend or os.environ.get(ENV_SIM_BACKEND)) or "isaacsim"
    normalized_compute = (
        normalize_compute_backend(compute_backend or os.environ.get(ENV_COMPUTE_BACKEND))
        or _default_compute_backend(device, normalized_sim)
    )
    return RuntimeSelection(
        compute_backend=normalized_compute,
        sim_backend=normalized_sim,
        device=device,
    )


def set_runtime_selection(runtime: RuntimeSelection) -> RuntimeSelection:
    """Persist the runtime selection in environment variables and builtins."""
    os.environ[ENV_COMPUTE_BACKEND] = runtime.compute_backend
    os.environ[ENV_SIM_BACKEND] = runtime.sim_backend
    builtins.ISAACLAB_COMPUTE_BACKEND = runtime.compute_backend
    builtins.ISAACLAB_SIM_BACKEND = runtime.sim_backend
    builtins.ISAACLAB_RUNTIME_DEVICE = runtime.device
    return runtime


def current_runtime(default_device: str = "cuda:0") -> RuntimeSelection:
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

    return RuntimeCapabilities(compute=compute, sim=sim)


def create_sim_backend(runtime: RuntimeSelection | None = None, *, simulation_context: Any | None = None) -> SimBackend:
    """Instantiate the simulator adapter for the active runtime."""
    runtime = runtime or current_runtime()
    if runtime.sim_backend == "isaacsim":
        return IsaacSimBackend(simulation_context=simulation_context)
    return MacSimBackend()


def get_runtime_state(runtime: RuntimeSelection | None = None) -> dict[str, Any]:
    """Return runtime selection plus capability metadata for diagnostics and tests."""
    runtime = runtime or current_runtime()
    capabilities = current_runtime_capabilities(runtime)
    return {
        "compute_backend": runtime.compute_backend,
        "sim_backend": runtime.sim_backend,
        "device": runtime.device,
        "capabilities": {
            "compute": capabilities.compute.__dict__,
            "sim": capabilities.sim.__dict__,
        },
    }


def require_runtime_backends(
    caller: str,
    *,
    compute_backend: ComputeBackendName | None = None,
    sim_backend: SimBackendName | None = None,
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
