# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLX/mac task registry surface."""

from __future__ import annotations

import importlib
import sys
import builtins
from pathlib import Path

import gymnasium as gym
import pytest

from isaaclab.backends import UnsupportedBackendError, resolve_runtime_selection, set_runtime_selection

SAFE_TASK_IDS = (
    "Isaac-Cartpole-RGB-Camera-Direct-v0",
    "Isaac-Cartpole-Depth-Camera-Direct-v0",
    "Isaac-Reach-Franka-v0",
    "Isaac-Reach-Franka-IK-Abs-v0",
    "Isaac-Reach-Franka-IK-Rel-v0",
    "Isaac-Reach-Franka-OSC-v0",
    "Isaac-Reach-OpenArm-v0",
    "Isaac-Reach-OpenArm-Bi-v0",
    "Isaac-Reach-UR10-v0",
    "Isaac-Deploy-Reach-UR10e-v0",
    "Isaac-Deploy-Reach-UR10e-Play-v0",
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Lift-Cube-Franka-IK-Abs-v0",
    "Isaac-Lift-Cube-Franka-IK-Rel-v0",
    "Isaac-Lift-Cube-OpenArm-v0",
    "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    "Isaac-Franka-Cabinet-Direct-v0",
    "Isaac-Open-Drawer-Franka-v0",
    "Isaac-Open-Drawer-Franka-IK-Abs-v0",
    "Isaac-Open-Drawer-OpenArm-v0",
    "Isaac-Velocity-Flat-H1-v0",
    "Isaac-Velocity-Rough-H1-v0",
    "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    "Isaac-Velocity-Rough-Anymal-C-Direct-v0",
    "Isaac-Cartpole-Direct-v0",
    "Isaac-Cart-Double-Pendulum-Direct-v0",
    "Isaac-Quadcopter-Direct-v0",
)
UNSUPPORTED_MAC_TASK_IDS = (
    "Isaac-Factory-PegInsert-Direct-v0",
    "Isaac-PickPlace-GR1T2-Abs-v0",
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0",
    "Isaac-Navigation-Flat-Anymal-C-v0",
    "Isaac-Tracking-LocoManip-Digit-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
)


def _clear_task_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "isaaclab_tasks" or module_name.startswith("isaaclab_tasks."):
            del sys.modules[module_name]


def _clear_task_specs() -> None:
    for task_id in SAFE_TASK_IDS + UNSUPPORTED_MAC_TASK_IDS:
        gym.registry.pop(task_id, None)


def test_mlx_task_registry_registers_supported_mac_tasks(monkeypatch):
    """The mac runtime should register supported and gated task ids without importing the full task tree."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    registry = importlib.import_module("isaaclab_tasks.registry")

    for task_id in SAFE_TASK_IDS:
        assert gym.spec(task_id).id == task_id

    openarm_reach_spec = gym.spec("Isaac-Reach-OpenArm-v0")
    openarm_bi_reach_spec = gym.spec("Isaac-Reach-OpenArm-Bi-v0")
    ur10_reach_spec = gym.spec("Isaac-Reach-UR10-v0")
    ur10e_spec = gym.spec("Isaac-Deploy-Reach-UR10e-v0")
    ur10e_play_spec = gym.spec("Isaac-Deploy-Reach-UR10e-Play-v0")
    openarm_lift_spec = gym.spec("Isaac-Lift-Cube-OpenArm-v0")
    openarm_open_drawer_spec = gym.spec("Isaac-Open-Drawer-OpenArm-v0")
    assert openarm_reach_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-openarm-surrogate"
    assert openarm_reach_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert openarm_bi_reach_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-openarm-bimanual-surrogate"
    assert openarm_bi_reach_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10_reach_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert ur10_reach_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10e_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert ur10e_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10e_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert ur10e_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert openarm_lift_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-openarm-surrogate"
    assert openarm_lift_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert openarm_open_drawer_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-openarm-surrogate"
    assert openarm_open_drawer_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False

    bin_stack_spec = gym.spec("Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0")
    assert bin_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-mimic"
    assert bin_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False

    for task_id in UNSUPPORTED_MAC_TASK_IDS:
        assert gym.spec(task_id).id == task_id


def test_parse_env_cfg_supports_mac_task_cfgs(monkeypatch):
    """parse_env_cfg should handle mac-native env config objects with root-level num_envs."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    cfg = parse_cfg.load_cfg_from_registry("Isaac-Cartpole-Direct-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Cartpole-Direct-v0", device="cpu", num_envs=32)

    assert type(cfg).__name__ == "MacCartpoleEnvCfg"
    assert parsed_cfg.num_envs == 32


def test_parse_env_cfg_supports_anymal_task_cfg(monkeypatch):
    """parse_env_cfg should resolve the mac-native ANYmal-C config without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    cfg = parse_cfg.load_cfg_from_registry("Isaac-Velocity-Flat-Anymal-C-Direct-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Velocity-Flat-Anymal-C-Direct-v0", device="cpu", num_envs=24)

    assert type(cfg).__name__ == "MacAnymalCFlatEnvCfg"
    assert parsed_cfg.num_envs == 24
    assert parsed_cfg.action_space == 12
    assert parsed_cfg.observation_space == 48


def test_parse_env_cfg_supports_anymal_rough_task_cfg(monkeypatch):
    """parse_env_cfg should resolve the mac-native rough ANYmal-C config without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    cfg = parse_cfg.load_cfg_from_registry("Isaac-Velocity-Rough-Anymal-C-Direct-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Velocity-Rough-Anymal-C-Direct-v0", device="cpu", num_envs=10)

    assert type(cfg).__name__ == "MacAnymalCRoughEnvCfg"
    assert parsed_cfg.num_envs == 10
    assert parsed_cfg.height_scan_enabled is True
    assert parsed_cfg.terrain_type == "wave"


def test_parse_env_cfg_supports_h1_task_cfg(monkeypatch):
    """parse_env_cfg should resolve the mac-native H1 config without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    cfg = parse_cfg.load_cfg_from_registry("Isaac-Velocity-Flat-H1-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Velocity-Flat-H1-v0", device="cpu", num_envs=12)

    assert type(cfg).__name__ == "MacH1FlatEnvCfg"
    assert parsed_cfg.num_envs == 12
    assert parsed_cfg.action_space == 19
    assert parsed_cfg.observation_space == 69


def test_parse_env_cfg_supports_h1_rough_task_cfg(monkeypatch):
    """parse_env_cfg should resolve the mac-native rough H1 config without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    cfg = parse_cfg.load_cfg_from_registry("Isaac-Velocity-Rough-H1-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Velocity-Rough-H1-v0", device="cpu", num_envs=9)

    assert type(cfg).__name__ == "MacH1RoughEnvCfg"
    assert parsed_cfg.num_envs == 9
    assert parsed_cfg.observation_space == 78
    assert parsed_cfg.height_scan_enabled is True
    assert parsed_cfg.terrain_type == "wave"


def test_parse_env_cfg_supports_franka_manipulation_task_cfgs(monkeypatch):
    """parse_env_cfg should resolve mac-native Franka manipulation configs without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    reach_cfg = parse_cfg.parse_env_cfg("Isaac-Reach-Franka-v0", device="cpu", num_envs=6)
    reach_ik_abs_cfg = parse_cfg.parse_env_cfg("Isaac-Reach-Franka-IK-Abs-v0", device="cpu", num_envs=7)
    reach_osc_cfg = parse_cfg.parse_env_cfg("Isaac-Reach-Franka-OSC-v0", device="cpu", num_envs=8)
    openarm_reach_cfg = parse_cfg.parse_env_cfg("Isaac-Reach-OpenArm-v0", device="cpu", num_envs=9)
    openarm_bi_reach_cfg = parse_cfg.parse_env_cfg("Isaac-Reach-OpenArm-Bi-v0", device="cpu", num_envs=10)
    ur10_reach_cfg = parse_cfg.parse_env_cfg("Isaac-Reach-UR10-v0", device="cpu", num_envs=11)
    lift_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cpu", num_envs=5)
    lift_ik_abs_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0", device="cpu", num_envs=6)
    lift_ik_rel_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Rel-v0", device="cpu", num_envs=7)
    openarm_lift_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Cube-OpenArm-v0", device="cpu", num_envs=8)
    teddy_bear_lift_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0", device="cpu", num_envs=8)
    stack_instance_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Instance-Randomize-Franka-v0", device="cpu", num_envs=9)
    stack_instance_ik_rel_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0", device="cpu", num_envs=10
    )
    stack_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-v0", device="cpu", num_envs=4)
    stack_ik_rel_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-v0", device="cpu", num_envs=9)
    stack_rgb_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0", device="cpu", num_envs=2)
    stack_rgb_alt_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0", device="cpu", num_envs=3)
    bin_stack_cfg_entry = parse_cfg.load_cfg_from_registry(
        "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0", "env_cfg_entry_point"
    )
    bin_stack_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0", device="cpu", num_envs=5)
    cabinet_cfg = parse_cfg.parse_env_cfg("Isaac-Franka-Cabinet-Direct-v0", device="cpu", num_envs=3)
    open_drawer_cfg = parse_cfg.parse_env_cfg("Isaac-Open-Drawer-Franka-IK-Abs-v0", device="cpu", num_envs=4)
    openarm_open_drawer_cfg = parse_cfg.parse_env_cfg("Isaac-Open-Drawer-OpenArm-v0", device="cpu", num_envs=5)

    assert type(reach_cfg).__name__ == "MacFrankaReachEnvCfg"
    assert reach_cfg.num_envs == 6
    assert reach_cfg.action_space == 7
    assert type(reach_ik_abs_cfg).__name__ == "MacFrankaReachEnvCfg"
    assert reach_ik_abs_cfg.num_envs == 7
    assert type(reach_osc_cfg).__name__ == "MacFrankaReachEnvCfg"
    assert reach_osc_cfg.num_envs == 8
    assert type(openarm_reach_cfg).__name__ == "MacOpenArmReachEnvCfg"
    assert openarm_reach_cfg.num_envs == 9
    assert openarm_reach_cfg.action_space == 7
    assert type(openarm_bi_reach_cfg).__name__ == "MacOpenArmBiReachEnvCfg"
    assert openarm_bi_reach_cfg.num_envs == 10
    assert openarm_bi_reach_cfg.action_space == 14
    assert openarm_bi_reach_cfg.observation_space == 46
    assert type(ur10_reach_cfg).__name__ == "MacUR10ReachEnvCfg"
    assert ur10_reach_cfg.num_envs == 11
    assert ur10_reach_cfg.action_space == 6
    assert type(lift_cfg).__name__ == "MacFrankaLiftEnvCfg"
    assert lift_cfg.num_envs == 5
    assert lift_cfg.action_space == 8
    assert type(lift_ik_abs_cfg).__name__ == "MacFrankaLiftEnvCfg"
    assert lift_ik_abs_cfg.num_envs == 6
    assert type(lift_ik_rel_cfg).__name__ == "MacFrankaLiftEnvCfg"
    assert lift_ik_rel_cfg.num_envs == 7
    assert type(openarm_lift_cfg).__name__ == "MacOpenArmLiftEnvCfg"
    assert openarm_lift_cfg.num_envs == 8
    assert openarm_lift_cfg.action_space == 8
    assert type(teddy_bear_lift_cfg).__name__ == "MacFrankaTeddyBearLiftEnvCfg"
    assert teddy_bear_lift_cfg.num_envs == 8
    assert teddy_bear_lift_cfg.action_space == 8
    assert type(stack_instance_cfg).__name__ == "MacFrankaStackInstanceRandomizeEnvCfg"
    assert stack_instance_cfg.num_envs == 9
    assert stack_instance_cfg.action_space == 8
    assert stack_instance_cfg.observation_space == 35
    assert type(stack_instance_ik_rel_cfg).__name__ == "MacFrankaStackInstanceRandomizeEnvCfg"
    assert stack_instance_ik_rel_cfg.num_envs == 10
    assert type(stack_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_cfg.num_envs == 4
    assert stack_cfg.action_space == 8
    assert type(stack_ik_rel_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_ik_rel_cfg.num_envs == 9
    assert type(stack_rgb_cfg).__name__ == "MacFrankaStackRgbEnvCfg"
    assert stack_rgb_cfg.num_envs == 2
    assert stack_rgb_cfg.action_space == 8
    assert stack_rgb_cfg.observation_space == 42
    assert type(stack_rgb_alt_cfg).__name__ == "MacFrankaStackRgbEnvCfg"
    assert stack_rgb_alt_cfg.num_envs == 3
    assert type(bin_stack_cfg_entry).__name__ == "MacFrankaBinStackEnvCfg"
    assert type(bin_stack_cfg).__name__ == "MacFrankaBinStackEnvCfg"
    assert bin_stack_cfg.num_envs == 5
    assert bin_stack_cfg.action_space == 8
    assert bin_stack_cfg.observation_space == 45
    assert bin_stack_cfg.semantic_contract == "reduced-no-mimic"
    assert bin_stack_cfg.upstream_alias_semantics_preserved is False
    assert "mimic" in bin_stack_cfg.contract_notes.lower()
    assert bin_stack_cfg.bin_anchor_observation_mode == "mirrored-support-anchor-tail"
    assert type(cabinet_cfg).__name__ == "MacFrankaCabinetEnvCfg"
    assert cabinet_cfg.num_envs == 3
    assert cabinet_cfg.action_space == 8
    assert type(open_drawer_cfg).__name__ == "MacFrankaOpenDrawerEnvCfg"
    assert open_drawer_cfg.num_envs == 4
    assert open_drawer_cfg.action_space == 8
    assert type(openarm_open_drawer_cfg).__name__ == "MacOpenArmOpenDrawerEnvCfg"
    assert openarm_open_drawer_cfg.num_envs == 5
    assert openarm_open_drawer_cfg.action_space == 8


def test_parse_env_cfg_supports_ur10e_deploy_reach_task_cfg(monkeypatch):
    """parse_env_cfg should resolve the reduced UR10e deploy-reach config on the mac path."""

    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")
    registry = importlib.import_module("isaaclab_tasks.registry")

    cfg = parse_cfg.parse_env_cfg("Isaac-Deploy-Reach-UR10e-v0", device="cpu", num_envs=6)
    play_cfg = parse_cfg.parse_env_cfg("Isaac-Deploy-Reach-UR10e-Play-v0", device="cpu", num_envs=7)
    spec = gym.spec("Isaac-Deploy-Reach-UR10e-v0")
    play_spec = gym.spec("Isaac-Deploy-Reach-UR10e-Play-v0")

    assert type(cfg).__name__ == "MacUR10eDeployReachEnvCfg"
    assert cfg.num_envs == 6
    assert cfg.action_space == 6
    assert cfg.observation_space == 19
    assert cfg.semantic_contract == "reduced-analytic-pose"
    assert cfg.upstream_alias_semantics_preserved is False
    assert type(play_cfg).__name__ == "MacUR10eDeployReachEnvCfg"
    assert play_cfg.num_envs == 7
    assert play_cfg.action_space == 6
    assert play_cfg.observation_space == 19
    assert play_cfg.semantic_contract == "reduced-analytic-pose"
    assert play_cfg.upstream_alias_semantics_preserved is False
    assert spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert play_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert play_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False


def test_parse_env_cfg_supports_cartpole_camera_task_cfgs(monkeypatch):
    """parse_env_cfg should resolve the synthetic camera cartpole configs without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    rgb_cfg = parse_cfg.parse_env_cfg("Isaac-Cartpole-RGB-Camera-Direct-v0", device="cpu", num_envs=4)
    depth_cfg = parse_cfg.parse_env_cfg("Isaac-Cartpole-Depth-Camera-Direct-v0", device="cpu", num_envs=3)

    assert type(rgb_cfg).__name__ == "MacCartpoleRGBCameraEnvCfg"
    assert rgb_cfg.num_envs == 4
    assert rgb_cfg.camera_mode == "rgb"
    assert rgb_cfg.observation_space == [100, 100, 3]
    assert type(depth_cfg).__name__ == "MacCartpoleDepthCameraEnvCfg"
    assert depth_cfg.num_envs == 3
    assert depth_cfg.camera_mode == "depth"
    assert depth_cfg.observation_space == [100, 100, 1]


def test_parse_env_cfg_reasserts_mac_safe_h1_registration_after_runtime_switch(monkeypatch):
    """H1 mac config loading should recover from package import before runtime selection."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()

    set_runtime_selection(resolve_runtime_selection(compute_backend="torch-cuda", sim_backend="isaacsim", device="cuda:0"))
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))
    cfg = parse_cfg.load_cfg_from_registry("Isaac-Velocity-Flat-H1-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Velocity-Flat-H1-v0", device="cpu", num_envs=20)

    assert type(cfg).__name__ == "MacH1FlatEnvCfg"
    assert parsed_cfg.num_envs == 20
    assert parsed_cfg.action_space == 19


def test_parse_env_cfg_keeps_mac_config_loading_free_of_mlx_runtime(monkeypatch):
    """Config parsing for mac tasks should not import MLX runtime modules."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("mlx"):
            raise ModuleNotFoundError("No module named 'mlx'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cfg = parse_cfg.parse_env_cfg("Isaac-Cartpole-Direct-v0", device="cpu", num_envs=16)

    assert type(cfg).__name__ == "MacCartpoleEnvCfg"
    assert cfg.num_envs == 16


def test_parse_env_cfg_rejects_isaacsim_only_tasks_on_mac(monkeypatch):
    """Isaac Sim-only tasks should fail with an explicit backend error on the mac runtime."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        parse_cfg.parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cpu")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        parse_cfg.parse_env_cfg("Isaac-Repose-Cube-Shadow-Vision-Direct-v0", device="cpu")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0", device="cpu")
