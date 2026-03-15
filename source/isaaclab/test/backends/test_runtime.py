# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for runtime backend resolution and device adapters."""

from __future__ import annotations

import argparse
import builtins
import importlib
import sys
import types

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

from isaaclab.app import AppLauncher
from isaaclab.backends import (
    CpuKernelBackend,
    IsaacSimBackend,
    IsaacSimPlannerBackend,
    IsaacSimSensorBackend,
    JointMotionPlanRequest,
    MlxComputeBackend,
    MacPlannerBackend,
    MacSensorBackend,
    MacSimBackend,
    MetalKernelBackend,
    PlannerWorldObstacle,
    PlannerWorldState,
    TorchCudaComputeBackend,
    ENV_COMPUTE_BACKEND,
    ENV_KERNEL_BACKEND,
    ENV_SIM_BACKEND,
    UnsupportedBackendError,
    UnsupportedRuntimeFeatureError,
    configure_torch_device,
    create_compute_backend,
    create_kernel_backend,
    create_planner_backend,
    create_sensor_backend,
    create_sim_backend,
    get_runtime_state,
    resolve_runtime_selection,
    set_runtime_selection,
    WarpKernelBackend,
)


@pytest.fixture(autouse=True)
def clear_runtime_env(monkeypatch: pytest.MonkeyPatch):
    """Keep runtime selection isolated per test."""
    monkeypatch.delenv(ENV_COMPUTE_BACKEND, raising=False)
    monkeypatch.delenv(ENV_KERNEL_BACKEND, raising=False)
    monkeypatch.delenv(ENV_SIM_BACKEND, raising=False)


def test_resolve_runtime_selection_from_environment(monkeypatch: pytest.MonkeyPatch):
    """Environment variables should seed the runtime selection."""
    monkeypatch.setenv(ENV_COMPUTE_BACKEND, "mlx")
    monkeypatch.setenv(ENV_KERNEL_BACKEND, "metal")
    monkeypatch.setenv(ENV_SIM_BACKEND, "mac-sim")

    runtime = resolve_runtime_selection(device="cpu")

    assert runtime.compute_backend == "mlx"
    assert runtime.sim_backend == "mac-sim"
    assert runtime.kernel_backend == "metal"
    assert runtime.device == "cpu"


def test_set_runtime_selection_persists_process_state():
    """The active runtime state should be exported for downstream consumers."""
    runtime = resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu")
    state = get_runtime_state(set_runtime_selection(runtime))

    assert state["compute_backend"] == "mlx"
    assert state["sim_backend"] == "mac-sim"
    assert state["kernel_backend"] == "metal"
    assert state["sensor_backend"] == "mac-sensors"
    assert state["planner_backend"] == "mac-planners"
    assert state["device"] == "cpu"
    assert state["supported_tasks"]["public_task_count"] >= 15


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
    assert "--kernel-backend" in parser._option_string_actions
    assert "--sim-backend" in parser._option_string_actions


def test_app_launcher_macsim_bootstrap_mode():
    """AppLauncher should support a mac-sim bootstrap path without requiring Isaac Sim modules."""
    launcher = AppLauncher(
        {
            "compute_backend": "mlx",
            "sim_backend": "mac-sim",
            "device": "cpu",
            "headless": True,
        }
    )

    assert launcher.sim_backend == "mac-sim"
    assert launcher.compute_backend == "mlx"
    assert launcher.kernel_backend == "metal"
    assert launcher.sensor_backend == "mac-sensors"
    assert launcher.planner_backend == "mac-planners"
    assert launcher.app.is_running() is False


def test_app_launcher_rejects_non_mlx_compute_for_macsim():
    """AppLauncher should reject unsupported compute backends for mac-sim."""
    with pytest.raises(UnsupportedBackendError, match="mac-sim"):
        AppLauncher(
            {
                "compute_backend": "torch-cuda",
                "sim_backend": "mac-sim",
                "device": "cuda:0",
                "headless": True,
            }
        )


def test_app_launcher_rejects_warp_kernel_for_macsim():
    """AppLauncher should reject Warp kernels on the mac-sim path."""
    with pytest.raises(UnsupportedBackendError, match="kernel-backend warp"):
        AppLauncher(
            {
                "compute_backend": "mlx",
                "sim_backend": "mac-sim",
                "kernel_backend": "warp",
                "device": "cpu",
                "headless": True,
            }
        )


def test_create_sim_backend_returns_isaacsim_adapter():
    """Isaac Sim runtime should produce the Isaac Sim adapter."""
    backend = create_sim_backend(resolve_runtime_selection("torch-cuda", "isaacsim", "cuda:0"))

    assert isinstance(backend, IsaacSimBackend)
    assert backend.contract.reset_signature == "reset(soft: bool = False) -> None"


def test_create_sim_backend_returns_macsim_adapter():
    """mac-sim runtime should produce the shared generic macOS adapter."""
    backend = create_sim_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert isinstance(backend, MacSimBackend)
    assert backend.contract.articulations.effort_targets is True
    state = backend.state_dict()
    assert state["implementation"] == "generic-articulation-layer+task-specialized-analytic-slices"
    assert state["generic_scene_runtime"] is True
    assert state["attached"] is False
    assert state["supported_tasks"]["current_mac_native_count"] >= 13


def test_create_kernel_backend_returns_warp_adapter():
    """Isaac Sim runtime should default to the Warp kernel adapter."""
    backend = create_kernel_backend(resolve_runtime_selection("torch-cuda", "isaacsim", "cuda:0"))

    assert isinstance(backend, WarpKernelBackend)
    assert backend.capabilities.custom_kernels is True


def test_create_kernel_backend_returns_metal_adapter():
    """mac-sim runtime should default to the Metal kernel adapter."""
    backend = create_kernel_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert isinstance(backend, MetalKernelBackend)
    assert backend.capabilities.raycast is True


def test_create_sensor_backend_follows_runtime():
    """Sensor backends should follow the selected simulation runtime."""
    upstream = create_sensor_backend(resolve_runtime_selection("torch-cuda", "isaacsim", "cuda:0"))
    mac = create_sensor_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert isinstance(upstream, IsaacSimSensorBackend)
    assert isinstance(mac, MacSensorBackend)
    assert mac.capabilities.raycast is True
    assert mac.capabilities.cameras is False
    assert mac.capabilities.depth is False
    assert mac.capabilities.rgb is False
    assert mac.capabilities.analytic_camera_tasks is True
    assert mac.capabilities.external_stereo_capture is True
    assert mac.state_dict()["implementation"] == "analytic-plane-raycast+synthetic-camera-tasks+external-stereo-depth-tooling"
    assert mac.state_dict()["camera_contract"] == "synthetic-task-slices+external-capture-only"


def test_create_planner_backend_follows_runtime():
    """Planner backends should follow the selected simulation runtime."""
    upstream = create_planner_backend(resolve_runtime_selection("torch-cuda", "isaacsim", "cuda:0"))
    mac = create_planner_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu"))

    assert isinstance(upstream, IsaacSimPlannerBackend)
    assert isinstance(mac, MacPlannerBackend)
    assert mac.capabilities.motion_generation is True
    assert mac.capabilities.batched_planning is True
    assert mac.capabilities.inverse_kinematics is False
    world_state = mac.update_world_state(
        PlannerWorldState(
            obstacles=(PlannerWorldObstacle("table", center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.1)),),
        )
    )
    request = JointMotionPlanRequest(
        joint_names=("joint_1", "joint_2"),
        start_positions=(0.0, -1.0),
        goal_positions=(1.0, 1.0),
        num_waypoints=5,
    )
    plan = mac.plan_joint_motion(request)

    assert world_state.state_dict()["obstacle_count"] == 1
    assert plan.waypoints[0] == (0.0, -1.0)
    assert plan.waypoints[-1] == (1.0, 1.0)
    assert plan.waypoint_times_s[-1] == pytest.approx(1.0)
    assert mac.state_dict()["implementation"] == "joint-space-linear-interpolation"


def test_create_cpu_kernel_backend():
    """Explicit CPU kernel selection should return the CPU backend."""
    backend = create_kernel_backend(resolve_runtime_selection("mlx", "mac-sim", "cpu", kernel_backend="cpu"))

    assert isinstance(backend, CpuKernelBackend)
    assert backend.capabilities.cpu_fallback is True


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


def test_mac_runtime_entrypoints_import_without_isaacsim():
    """The public mac entrypoints should import without requiring Isaac Sim modules."""
    require_mlx_runtime()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab")
    importlib.import_module("isaaclab.backends.runtime")
    importlib.import_module("isaaclab.backends.mac_sim")
    importlib.import_module("isaaclab.controllers")
    importlib.import_module("isaaclab.controllers.differential_ik_cfg")
    importlib.import_module("isaaclab.controllers.operational_space_cfg")
    importlib.import_module("isaaclab.sim")
    importlib.import_module("isaaclab.sim.schemas")
    importlib.import_module("isaaclab.sim.converters")
    importlib.import_module("isaaclab.sim.spawners.from_files")
    importlib.import_module("isaaclab.utils.nucleus")
    importlib.import_module("isaaclab.envs.mdp")
    importlib.import_module("isaaclab.envs.mdp.actions")
    importlib.import_module("isaaclab.envs.mdp.actions.actions_cfg")
    importlib.import_module("isaaclab.envs.mdp.actions.rmpflow_actions_cfg")
    importlib.import_module("isaaclab.controllers.rmp_flow_cfg")
    importlib.import_module("isaaclab.controllers.config.rmp_flow")

    from isaaclab.sim.spawners.from_files import GroundPlaneCfg

    cfg = GroundPlaneCfg()
    assert isinstance(cfg, GroundPlaneCfg)
    assert cfg.usd_path.endswith("default_environment.usd")


def test_mac_runtime_can_load_shared_config_helpers_without_torch(monkeypatch: pytest.MonkeyPatch):
    """The shared config/helper surface should stay import-safe without torch on the mac bootstrap path."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    sys.modules.pop("isaaclab.envs.common", None)
    sys.modules.pop("isaaclab.utils.noise", None)
    sys.modules.pop("isaaclab.utils.noise.noise_cfg", None)
    sys.modules.pop("isaaclab.utils.io", None)
    sys.modules.pop("isaaclab.utils.modifiers", None)
    sys.modules.pop("isaaclab.utils.modifiers.modifier_cfg_base", None)
    sys.modules.pop("isaaclab.utils.interpolation", None)
    sys.modules.pop("isaaclab.markers", None)
    sys.modules.pop("isaaclab.markers.config", None)
    sys.modules.pop("isaaclab.markers.visualization_markers_cfg", None)
    sys.modules.pop("isaaclab.devices.device_base", None)
    sys.modules.pop("isaaclab.devices.openxr", None)
    sys.modules.pop("isaaclab.devices.openxr.xr_cfg", None)
    sys.modules.pop("isaaclab.devices.openxr.retargeters", None)
    sys.modules.pop("isaaclab.devices.openxr.retargeters.manipulator", None)
    sys.modules.pop("isaaclab.sensors.camera", None)
    sys.modules.pop("isaaclab.sensors.camera.camera_cfg", None)
    sys.modules.pop("isaaclab.sensors.camera.tiled_camera_cfg", None)
    sys.modules.pop("isaaclab.sensors.ray_caster", None)
    sys.modules.pop("isaaclab.sensors.ray_caster.ray_caster_cfg", None)
    sys.modules.pop("isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg", None)
    sys.modules.pop("isaaclab.sensors.ray_caster.ray_caster_camera_cfg", None)
    sys.modules.pop("isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera_cfg", None)
    sys.modules.pop("isaaclab.sensors.ray_caster.patterns", None)
    sys.modules.pop("isaaclab.sensors.ray_caster.patterns.patterns_cfg", None)
    sys.modules.pop("isaaclab.sim.spawners.sensors", None)
    sys.modules.pop("isaaclab.sim.spawners.sensors.sensors_cfg", None)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = ("torch", "warp", "carb", "omni", "pxr")
        if name.startswith(blocked):
            raise ModuleNotFoundError(f"No module named '{name.split('.', 1)[0]}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    envs = importlib.import_module("isaaclab.envs")
    noise = importlib.import_module("isaaclab.utils.noise")
    io = importlib.import_module("isaaclab.utils.io")
    modifiers_mod = importlib.import_module("isaaclab.utils.modifiers")
    interpolation_mod = importlib.import_module("isaaclab.utils.interpolation")
    types_mod = importlib.import_module("isaaclab.utils.types")
    spaces = importlib.import_module("isaaclab.envs.utils.spaces")
    markers_mod = importlib.import_module("isaaclab.markers")
    markers_cfg = importlib.import_module("isaaclab.markers.config")
    openxr_mod = importlib.import_module("isaaclab.devices.openxr")
    retargeters_mod = importlib.import_module("isaaclab.devices.openxr.retargeters")
    manipulator_retargeters = importlib.import_module("isaaclab.devices.openxr.retargeters.manipulator")
    camera_mod = importlib.import_module("isaaclab.sensors.camera")
    ray_caster_mod = importlib.import_module("isaaclab.sensors.ray_caster")
    camera_cfg_mod = importlib.import_module("isaaclab.sensors.camera.camera_cfg")
    ray_caster_cfg_mod = importlib.import_module("isaaclab.sensors.ray_caster.ray_caster_cfg")
    ray_caster_camera_cfg_mod = importlib.import_module("isaaclab.sensors.ray_caster.ray_caster_camera_cfg")
    patterns_mod = importlib.import_module("isaaclab.sensors.ray_caster.patterns")
    sim_sensor_spawners = importlib.import_module("isaaclab.sim.spawners.sensors")

    assert envs.ViewerCfg.__name__ == "ViewerCfg"
    assert noise.NoiseModelCfg.__name__ == "NoiseModelCfg"
    assert callable(io.dump_yaml)
    assert modifiers_mod.ModifierCfg.__name__ == "ModifierCfg"
    assert interpolation_mod.__name__.endswith(".interpolation")
    assert types_mod.ArticulationActions.__name__ == "ArticulationActions"
    assert callable(spaces.spec_to_gym_space)
    assert markers_mod.VisualizationMarkersCfg.__name__ == "VisualizationMarkersCfg"
    assert markers_cfg.FRAME_MARKER_CFG.__class__.__name__ == "VisualizationMarkersCfg"
    assert openxr_mod.XrCfg.__name__ == "XrCfg"
    assert callable(openxr_mod.remove_camera_configs)
    assert retargeters_mod.__name__.endswith(".retargeters")
    assert manipulator_retargeters.__name__.endswith(".manipulator")
    assert camera_mod.CameraCfg.__name__ == "CameraCfg"
    assert camera_mod.TiledCameraCfg.__name__ == "TiledCameraCfg"
    assert camera_cfg_mod.CameraCfg.__name__ == "CameraCfg"
    assert ray_caster_mod.RayCasterCfg.__name__ == "RayCasterCfg"
    assert ray_caster_mod.MultiMeshRayCasterCfg.__name__ == "MultiMeshRayCasterCfg"
    assert ray_caster_cfg_mod.RayCasterCfg.__name__ == "RayCasterCfg"
    assert ray_caster_camera_cfg_mod.RayCasterCameraCfg.__name__ == "RayCasterCameraCfg"
    assert patterns_mod.PinholeCameraPatternCfg.__name__ == "PinholeCameraPatternCfg"
    assert sim_sensor_spawners.PinholeCameraCfg.__name__ == "PinholeCameraCfg"


def test_mac_sim_sensor_subpackages_gate_runtime_exports():
    """Sensor subpackages should import on mac-sim and fail explicitly when runtime exports are accessed."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    camera_mod = importlib.import_module("isaaclab.sensors.camera")
    ray_caster_mod = importlib.import_module("isaaclab.sensors.ray_caster")
    sim_sensor_spawners = importlib.import_module("isaaclab.sim.spawners.sensors")

    assert camera_mod.CameraCfg.__name__ == "CameraCfg"
    assert ray_caster_mod.RayCasterCfg.__name__ == "RayCasterCfg"
    assert sim_sensor_spawners.PinholeCameraCfg.__name__ == "PinholeCameraCfg"

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = camera_mod.Camera
    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = camera_mod.utils
    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = ray_caster_mod.RayCaster
    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = sim_sensor_spawners.spawn_camera


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
    require_mlx_runtime()
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


def test_macsim_backend_requires_scene_before_step():
    """mac-sim adapter should fail explicitly until a generic scene substrate is attached."""
    backend = MacSimBackend()

    with pytest.raises(UnsupportedRuntimeFeatureError, match="attached generic scene substrate"):
        backend.step()


def test_macsim_backend_proxies_generic_scene_and_articulation_calls():
    """mac-sim should expose generic batched articulation/root-state IO through the shared scene substrate."""
    require_mlx_runtime()

    from isaaclab.backends.mac_sim import EnvironmentOriginGrid, MacSimSceneState

    backend = MacSimBackend()
    scene = backend.create_scene(num_envs=3, physics_dt=0.02)
    origin_grid = EnvironmentOriginGrid(num_envs=3, spacing=1.5)
    articulation = scene.add_articulation(
        "robot",
        num_joints=2,
        with_root_state=True,
        origin_grid=origin_grid,
        default_joint_pos=(0.1, -0.1),
        default_root_pose=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
    )

    assert isinstance(scene, MacSimSceneState)
    joint_pos, joint_vel = backend.get_joint_state("robot")
    assert joint_pos.shape == (3, 2)
    assert joint_vel.shape == (3, 2)

    backend.set_joint_effort_target("robot", [0.25, -0.25, 0.5], joint_ids=[0])
    backend.write_joint_state("robot", [[1.0, 2.0]], [[0.1, 0.2]], env_ids=[1])
    backend.write_root_pose("robot", [[0.2, 0.3, 0.8, 0.0, 0.0, 0.0, 1.0]], env_ids=[2])
    backend.write_root_velocity("robot", [[0.4, 0.0, -0.1, 0.01, 0.02, 0.03]], env_ids=[2])
    step_state = backend.step(render=False, update_fabric=True)

    assert step_state["step_count"] == 1
    assert step_state["last_step_args"] == (False, True)
    assert articulation.joint_state.joint_effort_target[:, 0].shape == (3,)
    assert articulation.root_state is not None

    reset_state = backend.reset()
    assert reset_state["step_count"] == 0
    assert reset_state["reset_count"] == 1
    backend_state = backend.state_dict()
    assert backend_state["attached"] is True
    assert backend_state["scene_state"]["articulation_count"] == 1


def test_mac_sim_env_exports_fail_with_clear_backend_error():
    """Import-time env access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    envs = importlib.import_module("isaaclab.envs")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = envs.DirectRLEnv


def test_mac_sim_marl_helpers_fail_with_clear_backend_error():
    """Multi-agent conversion helpers should stay gated behind the Isaac Sim runtime."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    envs = importlib.import_module("isaaclab.envs")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = envs.multi_agent_to_single_agent


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


def test_mac_sim_sensor_exports_fail_with_clear_backend_error():
    """Import-time sensor access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    sensors = importlib.import_module("isaaclab.sensors")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = sensors.Camera


def test_mac_sim_manager_exports_fail_with_clear_backend_error():
    """Import-time manager access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    managers = importlib.import_module("isaaclab.managers")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = managers.ActionManager


def test_mac_sim_controller_exports_fail_with_clear_backend_error():
    """Import-time controller access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    controllers = importlib.import_module("isaaclab.controllers")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = controllers.DifferentialIKController


def test_mac_sim_device_exports_fail_with_clear_backend_error():
    """Import-time device access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    devices = importlib.import_module("isaaclab.devices")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = devices.Se2Keyboard


def test_mac_sim_scene_exports_fail_with_clear_backend_error():
    """Import-time scene access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    scene = importlib.import_module("isaaclab.scene")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = scene.InteractiveScene


def test_mac_sim_marker_exports_fail_with_clear_backend_error():
    """Import-time marker access on mac-sim should raise a backend error instead of importing Omniverse modules."""
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    markers = importlib.import_module("isaaclab.markers")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        _ = markers.VisualizationMarkers
