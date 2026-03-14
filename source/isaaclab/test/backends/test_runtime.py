# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for runtime backend resolution and device adapters."""

from __future__ import annotations

import argparse
import importlib
import sys
import types

import pytest

from isaaclab.app import AppLauncher
from isaaclab.backends import (
    IsaacSimBackend,
    MlxComputeBackend,
    MacSimBackend,
    TorchCudaComputeBackend,
    ENV_COMPUTE_BACKEND,
    ENV_SIM_BACKEND,
    UnsupportedBackendError,
    UnsupportedRuntimeFeatureError,
    configure_torch_device,
    create_compute_backend,
    create_sim_backend,
    get_runtime_state,
    resolve_runtime_selection,
    set_runtime_selection,
)


@pytest.fixture(autouse=True)
def clear_runtime_env(monkeypatch: pytest.MonkeyPatch):
    """Keep runtime selection isolated per test."""
    monkeypatch.delenv(ENV_COMPUTE_BACKEND, raising=False)
    monkeypatch.delenv(ENV_SIM_BACKEND, raising=False)


def test_resolve_runtime_selection_from_environment(monkeypatch: pytest.MonkeyPatch):
    """Environment variables should seed the runtime selection."""
    monkeypatch.setenv(ENV_COMPUTE_BACKEND, "mlx")
    monkeypatch.setenv(ENV_SIM_BACKEND, "mac-sim")

    runtime = resolve_runtime_selection(device="cpu")

    assert runtime.compute_backend == "mlx"
    assert runtime.sim_backend == "mac-sim"
    assert runtime.device == "cpu"


def test_set_runtime_selection_persists_process_state():
    """The active runtime state should be exported for downstream consumers."""
    runtime = resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu")
    state = get_runtime_state(set_runtime_selection(runtime))

    assert state["compute_backend"] == "mlx"
    assert state["sim_backend"] == "mac-sim"
    assert state["device"] == "cpu"


def test_configure_torch_device_skips_cuda_for_mlx(monkeypatch: pytest.MonkeyPatch):
    """MLX mode must not touch torch CUDA device selection."""
    runtime = set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    calls: list[str] = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda device: calls.append(device)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    configure_torch_device("cuda:3", runtime)

    assert calls == []


def test_configure_torch_device_routes_cuda_for_torch_backend(monkeypatch: pytest.MonkeyPatch):
    """Torch runtime should still configure CUDA devices through the adapter."""
    runtime = set_runtime_selection(
        resolve_runtime_selection(compute_backend="torch-cuda", sim_backend="isaacsim", device="cuda:1")
    )
    calls: list[str] = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda device: calls.append(device)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    configure_torch_device("cuda:1", runtime)

    assert calls == ["cuda:1"]


def test_app_launcher_arg_parser_exposes_backend_flags():
    """AppLauncher should publish the backend selection CLI knobs."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    assert "--compute-backend" in parser._option_string_actions
    assert "--sim-backend" in parser._option_string_actions


def test_create_sim_backend_returns_isaacsim_adapter():
    """Isaac Sim runtime should produce the Isaac Sim adapter."""
    backend = create_sim_backend(resolve_runtime_selection("torch-cuda", "isaacsim", "cuda:0"))

    assert isinstance(backend, IsaacSimBackend)
    assert backend.contract.reset_signature == "reset(soft: bool = False) -> None"


def test_create_sim_backend_returns_macsim_adapter():
    """mac-sim runtime should produce the placeholder macOS adapter."""
    backend = create_sim_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert isinstance(backend, MacSimBackend)
    assert backend.contract.articulations.effort_targets is False


def test_create_compute_backend_returns_torch_adapter():
    """Torch runtime should produce the torch compute adapter."""
    backend = create_compute_backend(resolve_runtime_selection("torch-cuda", "isaacsim", "cuda:0"))

    assert isinstance(backend, TorchCudaComputeBackend)
    assert backend.capabilities.torch_interop is True


def test_create_compute_backend_returns_mlx_adapter():
    """MLX runtime should produce the MLX compute adapter."""
    backend = create_compute_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert isinstance(backend, MlxComputeBackend)
    assert backend.capabilities.torch_interop is False


def test_torch_compute_backend_routes_device_seed_and_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Torch compute adapter should forward device, seed, and checkpoint I/O to torch."""
    calls: list[tuple[str, object]] = []
    payload = {"value": 123}

    class FakeTorch:
        class cuda:
            @staticmethod
            def set_device(device):
                calls.append(("set_device", device))

            @staticmethod
            def manual_seed_all(seed):
                calls.append(("manual_seed_all", seed))

        @staticmethod
        def manual_seed(seed):
            calls.append(("manual_seed", seed))

        @staticmethod
        def save(data, path):
            calls.append(("save", path))
            path.write_text(str(data), encoding="utf-8")

        @staticmethod
        def load(path, map_location=None):
            calls.append(("load", (path, map_location)))
            return {"loaded": path.read_text(encoding="utf-8")}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    backend = TorchCudaComputeBackend()
    checkpoint = tmp_path / "torch_ckpt.txt"

    backend.configure_device("cuda:1")
    backend.seed(9)
    backend.save_checkpoint(checkpoint, payload)
    loaded = backend.load_checkpoint(checkpoint)

    assert calls[0] == ("set_device", "cuda:1")
    assert ("manual_seed", 9) in calls
    assert ("manual_seed_all", 9) in calls
    assert ("save", checkpoint) in calls
    assert ("load", (checkpoint, "cpu")) in calls
    assert loaded == {"loaded": str(payload)}


def test_mlx_compute_backend_seed_and_checkpoint(tmp_path):
    """MLX compute adapter should seed the backend and round-trip checkpoints."""
    pytest.importorskip("mlx.core")
    backend = MlxComputeBackend()
    checkpoint = tmp_path / "mlx_ckpt.pkl"
    payload = {"episode": 4, "reward": 1.5}

    backend.seed(17)
    backend.save_checkpoint(str(checkpoint), payload)
    loaded = backend.load_checkpoint(str(checkpoint))

    assert loaded == payload


def test_isaacsim_backend_proxies_simulation_and_articulation_calls():
    """IsaacSim backend should mirror the upstream simulation/articulation surface used by cartpole."""
    calls: list[tuple[str, object]] = []

    class FakeSimulationContext:
        device = "cuda:0"
        render_mode = types.SimpleNamespace(name="NO_RENDERING")

        def reset(self, *, soft: bool = False):
            calls.append(("reset", soft))

        def step(self, *, render: bool = True, update_fabric: bool = False):
            calls.append(("step", (render, update_fabric)))

    class FakeArticulation:
        def __init__(self):
            self.data = types.SimpleNamespace(joint_pos="joint-pos", joint_vel="joint-vel")

        def set_joint_effort_target(self, efforts, joint_ids=None):
            calls.append(("effort", (efforts, joint_ids)))

        def write_joint_state_to_sim(self, joint_pos, joint_vel, joint_acc, env_ids):
            calls.append(("joint-state", (joint_pos, joint_vel, joint_acc, env_ids)))

        def write_root_pose_to_sim(self, root_pose, env_ids):
            calls.append(("root-pose", (root_pose, env_ids)))

        def write_root_velocity_to_sim(self, root_velocity, env_ids):
            calls.append(("root-velocity", (root_velocity, env_ids)))

    backend = IsaacSimBackend(FakeSimulationContext())
    articulation = FakeArticulation()

    assert backend.get_joint_state(articulation) == ("joint-pos", "joint-vel")
    backend.reset(soft=True)
    backend.step(render=False, update_fabric=True)
    backend.set_joint_effort_target(articulation, "efforts", joint_ids=[0])
    backend.write_joint_state(articulation, "pos", "vel", joint_acc=None, env_ids=[1])
    backend.write_root_pose(articulation, "pose", env_ids=[1])
    backend.write_root_velocity(articulation, "velocity", env_ids=[1])

    assert calls == [
        ("reset", True),
        ("step", (False, True)),
        ("effort", ("efforts", [0])),
        ("joint-state", ("pos", "vel", None, [1])),
        ("root-pose", ("pose", [1])),
        ("root-velocity", ("velocity", [1])),
    ]
    assert backend.state_dict()["attached"] is True


def test_macsim_backend_raises_clear_unimplemented_errors():
    """mac-sim adapter should fail explicitly until the simulator implementation exists."""
    backend = MacSimBackend()

    with pytest.raises(UnsupportedRuntimeFeatureError, match="mac-sim"):
        backend.step()


def test_mac_sim_env_exports_fail_with_clear_backend_error():
    """Import-time env access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    envs = importlib.import_module("isaaclab.envs")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = envs.DirectRLEnv


def test_mac_sim_can_load_simulation_cfg_but_not_simulation_context():
    """The mac-sim bootstrap path should expose config objects without exposing Isaac Sim runtime objects."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    sim = importlib.import_module("isaaclab.sim")

    assert sim.SimulationCfg.__name__ == "SimulationCfg"
    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = sim.SimulationContext


def test_mac_sim_asset_exports_fail_with_clear_backend_error():
    """Import-time asset access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    assets = importlib.import_module("isaaclab.assets")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = assets.Articulation
