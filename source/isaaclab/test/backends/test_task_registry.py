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
    "Isaac-Deploy-Reach-UR10e-ROS-Inference-v0",
    "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
    "Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0",
    "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0",
    "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
    "Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0",
    "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0",
    "Isaac-Factory-PegInsert-Direct-v0",
    "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
    "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0",
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Lift-Cube-Franka-IK-Abs-v0",
    "Isaac-Lift-Cube-Franka-IK-Rel-v0",
    "Isaac-Lift-Cube-Franka-IK-Rel-Play-v0",
    "Isaac-Lift-Cube-OpenArm-v0",
    "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
    "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-Play-v0",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0",
    "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Play-v0",
    "Isaac-Stack-Cube-Franka-IK-Abs-Play-v0",
    "Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-Play-v0",
    "Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-Play-v0",
    "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-Play-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    "Isaac-PickPlace-GR1T2-Abs-v0",
    "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
    "Isaac-PickPlace-G1-InspireFTP-Abs-v0",
    "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
    "Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
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
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    "Isaac-Navigation-Flat-Anymal-C-v0",
    "Isaac-Tracking-LocoManip-Digit-v0",
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
    ur10e_ros_inference_spec = gym.spec("Isaac-Deploy-Reach-UR10e-ROS-Inference-v0")
    ur10e_gear_2f140_spec = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F140-v0")
    ur10e_gear_2f140_play_spec = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0")
    ur10e_gear_2f85_spec = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F85-v0")
    ur10e_gear_2f85_play_spec = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0")
    factory_peg_insert_spec = gym.spec("Isaac-Factory-PegInsert-Direct-v0")
    openarm_lift_spec = gym.spec("Isaac-Lift-Cube-OpenArm-v0")
    agibot_toy2box_spec = gym.spec("Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0")
    agibot_mug_spec = gym.spec("Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0")
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
    assert ur10e_ros_inference_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-ros-inference"
    assert ur10e_ros_inference_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10e_gear_2f140_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert ur10e_gear_2f140_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10e_gear_2f140_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert ur10e_gear_2f140_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10e_gear_2f85_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert ur10e_gear_2f85_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ur10e_gear_2f85_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert ur10e_gear_2f85_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert factory_peg_insert_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-peg-insert"
    assert factory_peg_insert_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert openarm_lift_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-openarm-surrogate"
    assert openarm_lift_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert agibot_toy2box_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-agibot-place-surrogate"
    assert agibot_toy2box_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert agibot_mug_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-agibot-place-surrogate"
    assert agibot_mug_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert openarm_open_drawer_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-openarm-surrogate"
    assert openarm_open_drawer_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False

    bin_stack_spec = gym.spec("Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0")
    stack_ik_rel_play_spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-Play-v0")
    blueprint_stack_spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0")
    skillgen_stack_spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0")
    visuomotor_stack_spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0")
    visuomotor_cosmos_stack_spec = gym.spec("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0")
    assert bin_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-mimic"
    assert bin_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert stack_ik_rel_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-stack"
    assert stack_ik_rel_play_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert blueprint_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-blueprint"
    assert blueprint_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert skillgen_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-skillgen"
    assert skillgen_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert visuomotor_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-visuomotor-surrogate"
    assert visuomotor_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert visuomotor_cosmos_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-cosmos"
    assert visuomotor_cosmos_stack_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False

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


def test_parse_env_cfg_supports_factory_peg_insert_task_cfg(monkeypatch):
    """parse_env_cfg should resolve the reduced factory peg-insert config without Isaac Sim imports."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")

    cfg = parse_cfg.load_cfg_from_registry("Isaac-Factory-PegInsert-Direct-v0", "env_cfg_entry_point")
    parsed_cfg = parse_cfg.parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cpu", num_envs=16)

    assert type(cfg).__name__ == "MacFactoryPegInsertEnvCfg"
    assert type(parsed_cfg).__name__ == "MacFactoryPegInsertEnvCfg"
    assert parsed_cfg.num_envs == 16
    assert parsed_cfg.semantic_contract == "reduced-analytic-peg-insert"
    assert parsed_cfg.upstream_alias_semantics_preserved is False
    assert parsed_cfg.gripper_variant == "peg-insert"


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
    lift_ik_rel_play_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Rel-Play-v0", device="cpu", num_envs=8)
    openarm_lift_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Cube-OpenArm-v0", device="cpu", num_envs=8)
    teddy_bear_lift_cfg = parse_cfg.parse_env_cfg("Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0", device="cpu", num_envs=8)
    stack_instance_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Instance-Randomize-Franka-v0", device="cpu", num_envs=9)
    stack_instance_play_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-Instance-Randomize-Franka-Play-v0", device="cpu", num_envs=9
    )
    stack_instance_ik_abs_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0", device="cpu", num_envs=10
    )
    stack_instance_ik_rel_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0", device="cpu", num_envs=10
    )
    stack_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-v0", device="cpu", num_envs=4)
    stack_ik_rel_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-v0", device="cpu", num_envs=9)
    stack_ik_abs_play_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Abs-Play-v0", device="cpu", num_envs=10)
    stack_ik_rel_play_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Rel-Play-v0", device="cpu", num_envs=10
    )
    stack_redgreen_play_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-Play-v0", device="cpu", num_envs=4
    )
    stack_bluegreen_play_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-Play-v0", device="cpu", num_envs=4
    )
    stack_blueprint_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0", device="cpu", num_envs=6)
    stack_skillgen_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0", device="cpu", num_envs=7)
    stack_rgb_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0", device="cpu", num_envs=2)
    stack_rgb_alt_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0", device="cpu", num_envs=3)
    stack_rgb_alt_play_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-Play-v0", device="cpu", num_envs=3
    )
    stack_visuomotor_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0", device="cpu", num_envs=4)
    stack_visuomotor_cosmos_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0", device="cpu", num_envs=5
    )
    bin_stack_cfg_entry = parse_cfg.load_cfg_from_registry(
        "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0", "env_cfg_entry_point"
    )
    bin_stack_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0", device="cpu", num_envs=5)
    pick_place_base_cfg = parse_cfg.parse_env_cfg("Isaac-PickPlace-GR1T2-Abs-v0", device="cpu", num_envs=4)
    pick_place_cfg = parse_cfg.parse_env_cfg("Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0", device="cpu", num_envs=5)
    agibot_toy2box_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0", device="cpu", num_envs=6
    )
    agibot_upright_mug_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0", device="cpu", num_envs=7
    )
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
    assert type(lift_ik_rel_play_cfg).__name__ == "MacFrankaLiftEnvCfg"
    assert lift_ik_rel_play_cfg.num_envs == 8
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
    assert type(stack_instance_play_cfg).__name__ == "MacFrankaStackInstanceRandomizeEnvCfg"
    assert stack_instance_play_cfg.num_envs == 9
    assert type(stack_instance_ik_abs_cfg).__name__ == "MacFrankaStackInstanceRandomizeEnvCfg"
    assert stack_instance_ik_abs_cfg.num_envs == 10
    assert type(stack_instance_ik_rel_cfg).__name__ == "MacFrankaStackInstanceRandomizeEnvCfg"
    assert stack_instance_ik_rel_cfg.num_envs == 10
    assert type(stack_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_cfg.num_envs == 4
    assert stack_cfg.action_space == 8
    assert type(stack_ik_rel_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_ik_rel_cfg.num_envs == 9
    assert type(stack_ik_abs_play_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_ik_abs_play_cfg.num_envs == 10
    assert type(stack_ik_rel_play_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_ik_rel_play_cfg.num_envs == 10
    assert stack_ik_rel_play_cfg.action_space == 8
    assert stack_ik_rel_play_cfg.semantic_contract == "reduced-analytic-stack"
    assert stack_ik_rel_play_cfg.upstream_alias_semantics_preserved is False
    assert type(stack_redgreen_play_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_redgreen_play_cfg.num_envs == 4
    assert type(stack_bluegreen_play_cfg).__name__ == "MacFrankaStackEnvCfg"
    assert stack_bluegreen_play_cfg.num_envs == 4
    assert type(stack_blueprint_cfg).__name__ == "MacFrankaStackBlueprintEnvCfg"
    assert stack_blueprint_cfg.num_envs == 6
    assert stack_blueprint_cfg.semantic_contract == "reduced-no-blueprint"
    assert stack_blueprint_cfg.upstream_alias_semantics_preserved is False
    assert type(stack_skillgen_cfg).__name__ == "MacFrankaStackSkillgenEnvCfg"
    assert stack_skillgen_cfg.num_envs == 7
    assert stack_skillgen_cfg.semantic_contract == "reduced-no-skillgen"
    assert stack_skillgen_cfg.upstream_alias_semantics_preserved is False
    assert type(stack_rgb_cfg).__name__ == "MacFrankaStackRgbEnvCfg"
    assert stack_rgb_cfg.num_envs == 2
    assert stack_rgb_cfg.action_space == 8
    assert stack_rgb_cfg.observation_space == 42
    assert type(stack_rgb_alt_cfg).__name__ == "MacFrankaStackRgbEnvCfg"
    assert stack_rgb_alt_cfg.num_envs == 3
    assert type(stack_rgb_alt_play_cfg).__name__ == "MacFrankaStackRgbEnvCfg"
    assert stack_rgb_alt_play_cfg.num_envs == 3
    assert type(stack_visuomotor_cfg).__name__ == "MacFrankaStackVisuomotorEnvCfg"
    assert stack_visuomotor_cfg.num_envs == 4
    assert stack_visuomotor_cfg.action_space == 8
    assert stack_visuomotor_cfg.observation_space == 42
    assert stack_visuomotor_cfg.semantic_contract == "reduced-visuomotor-surrogate"
    assert stack_visuomotor_cfg.upstream_alias_semantics_preserved is False
    assert type(stack_visuomotor_cosmos_cfg).__name__ == "MacFrankaStackVisuomotorCosmosEnvCfg"
    assert stack_visuomotor_cosmos_cfg.num_envs == 5
    assert stack_visuomotor_cosmos_cfg.action_space == 8
    assert stack_visuomotor_cosmos_cfg.observation_space == 42
    assert stack_visuomotor_cosmos_cfg.semantic_contract == "reduced-no-cosmos"
    assert stack_visuomotor_cosmos_cfg.upstream_alias_semantics_preserved is False
    assert type(bin_stack_cfg_entry).__name__ == "MacFrankaBinStackEnvCfg"
    assert type(bin_stack_cfg).__name__ == "MacFrankaBinStackEnvCfg"
    assert bin_stack_cfg.num_envs == 5
    assert bin_stack_cfg.action_space == 8
    assert bin_stack_cfg.observation_space == 45
    assert bin_stack_cfg.semantic_contract == "reduced-no-mimic"
    assert bin_stack_cfg.upstream_alias_semantics_preserved is False
    assert "mimic" in bin_stack_cfg.contract_notes.lower()
    assert bin_stack_cfg.bin_anchor_observation_mode == "mirrored-support-anchor-tail"
    assert type(pick_place_base_cfg).__name__ == "MacFrankaBinStackPickPlaceEnvCfg"
    assert pick_place_base_cfg.num_envs == 4
    assert pick_place_base_cfg.action_space == 8
    assert pick_place_base_cfg.observation_space == 45
    assert pick_place_base_cfg.semantic_contract == "reduced-pick-place-surrogate"
    assert pick_place_base_cfg.upstream_alias_semantics_preserved is False
    assert "pick-place" in pick_place_base_cfg.contract_notes.lower()
    assert type(pick_place_cfg).__name__ == "MacFrankaBinStackPickPlaceEnvCfg"
    assert pick_place_cfg.num_envs == 5
    assert pick_place_cfg.action_space == 8
    assert pick_place_cfg.observation_space == 45
    assert pick_place_cfg.semantic_contract == "reduced-pick-place-surrogate"
    assert pick_place_cfg.upstream_alias_semantics_preserved is False
    assert "pick-place" in pick_place_cfg.contract_notes.lower()
    assert type(agibot_toy2box_cfg).__name__ == "MacAgibotPlaceToy2BoxEnvCfg"
    assert agibot_toy2box_cfg.num_envs == 6
    assert agibot_toy2box_cfg.action_space == 8
    assert agibot_toy2box_cfg.observation_space == 34
    assert agibot_toy2box_cfg.semantic_contract == "reduced-agibot-place-surrogate"
    assert agibot_toy2box_cfg.upstream_alias_semantics_preserved is False
    assert "agibot" in agibot_toy2box_cfg.contract_notes.lower()
    assert type(agibot_upright_mug_cfg).__name__ == "MacAgibotPlaceUprightMugEnvCfg"
    assert agibot_upright_mug_cfg.num_envs == 7
    assert agibot_upright_mug_cfg.action_space == 8
    assert agibot_upright_mug_cfg.observation_space == 34
    assert agibot_upright_mug_cfg.semantic_contract == "reduced-agibot-place-surrogate"
    assert agibot_upright_mug_cfg.upstream_alias_semantics_preserved is False
    assert "agibot" in agibot_upright_mug_cfg.contract_notes.lower()
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
    ros_inference_cfg = parse_cfg.parse_env_cfg("Isaac-Deploy-Reach-UR10e-ROS-Inference-v0", device="cpu", num_envs=8)
    spec = gym.spec("Isaac-Deploy-Reach-UR10e-v0")
    play_spec = gym.spec("Isaac-Deploy-Reach-UR10e-Play-v0")
    ros_inference_spec = gym.spec("Isaac-Deploy-Reach-UR10e-ROS-Inference-v0")

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
    assert type(ros_inference_cfg).__name__ == "MacUR10eDeployReachRosInferenceEnvCfg"
    assert ros_inference_cfg.num_envs == 8
    assert ros_inference_cfg.action_space == 6
    assert ros_inference_cfg.observation_space == 19
    assert ros_inference_cfg.semantic_contract == "reduced-no-ros-inference"
    assert ros_inference_cfg.upstream_alias_semantics_preserved is False
    assert spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert play_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-pose"
    assert play_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ros_inference_spec.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-ros-inference"
    assert ros_inference_spec.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False


def test_parse_env_cfg_supports_ur10e_gear_assembly_task_cfgs(monkeypatch):
    """parse_env_cfg should resolve the reduced UR10e gear-assembly configs on the mac path."""

    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")
    registry = importlib.import_module("isaaclab_tasks.registry")

    cfg_2f140 = parse_cfg.parse_env_cfg("Isaac-Deploy-GearAssembly-UR10e-2F140-v0", device="cpu", num_envs=6)
    play_cfg_2f140 = parse_cfg.parse_env_cfg("Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0", device="cpu", num_envs=7)
    ros_cfg_2f140 = parse_cfg.parse_env_cfg(
        "Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0", device="cpu", num_envs=10
    )
    cfg_2f85 = parse_cfg.parse_env_cfg("Isaac-Deploy-GearAssembly-UR10e-2F85-v0", device="cpu", num_envs=8)
    play_cfg_2f85 = parse_cfg.parse_env_cfg("Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0", device="cpu", num_envs=9)
    ros_cfg_2f85 = parse_cfg.parse_env_cfg(
        "Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0", device="cpu", num_envs=11
    )
    long_stack_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0", device="cpu", num_envs=12)
    short_stack_cfg = parse_cfg.parse_env_cfg(
        "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0", device="cpu", num_envs=13
    )
    spec_2f140 = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F140-v0")
    play_spec_2f140 = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0")
    ros_spec_2f140 = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0")
    spec_2f85 = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F85-v0")
    play_spec_2f85 = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0")
    ros_spec_2f85 = gym.spec("Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0")
    long_stack_spec = gym.spec("Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0")
    short_stack_spec = gym.spec("Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0")

    assert type(cfg_2f140).__name__ == "MacUR10eGearAssembly2F140EnvCfg"
    assert cfg_2f140.num_envs == 6
    assert cfg_2f140.action_space == 6
    assert cfg_2f140.observation_space == 19
    assert cfg_2f140.semantic_contract == "reduced-analytic-assembly"
    assert cfg_2f140.upstream_alias_semantics_preserved is False
    assert cfg_2f140.gripper_variant == "2f140"
    assert type(play_cfg_2f140).__name__ == "MacUR10eGearAssembly2F140EnvCfg"
    assert play_cfg_2f140.num_envs == 7
    assert play_cfg_2f140.action_space == 6
    assert play_cfg_2f140.observation_space == 19
    assert type(ros_cfg_2f140).__name__ == "MacUR10eGearAssembly2F140RosInferenceEnvCfg"
    assert ros_cfg_2f140.num_envs == 10
    assert ros_cfg_2f140.action_space == 6
    assert ros_cfg_2f140.observation_space == 19
    assert ros_cfg_2f140.semantic_contract == "reduced-no-ros-inference"
    assert ros_cfg_2f140.upstream_alias_semantics_preserved is False
    assert type(cfg_2f85).__name__ == "MacUR10eGearAssembly2F85EnvCfg"
    assert cfg_2f85.num_envs == 8
    assert cfg_2f85.action_space == 6
    assert cfg_2f85.observation_space == 19
    assert cfg_2f85.semantic_contract == "reduced-analytic-assembly"
    assert cfg_2f85.upstream_alias_semantics_preserved is False
    assert cfg_2f85.gripper_variant == "2f85"
    assert type(play_cfg_2f85).__name__ == "MacUR10eGearAssembly2F85EnvCfg"
    assert play_cfg_2f85.num_envs == 9
    assert play_cfg_2f85.action_space == 6
    assert play_cfg_2f85.observation_space == 19
    assert type(ros_cfg_2f85).__name__ == "MacUR10eGearAssembly2F85RosInferenceEnvCfg"
    assert ros_cfg_2f85.num_envs == 11
    assert ros_cfg_2f85.action_space == 6
    assert ros_cfg_2f85.observation_space == 19
    assert ros_cfg_2f85.semantic_contract == "reduced-no-ros-inference"
    assert ros_cfg_2f85.upstream_alias_semantics_preserved is False
    assert type(long_stack_cfg).__name__ == "MacUR10LongSuctionStackEnvCfg"
    assert long_stack_cfg.num_envs == 12
    assert long_stack_cfg.action_space == 7
    assert long_stack_cfg.observation_space == 40
    assert long_stack_cfg.semantic_contract == "reduced-analytic-suction-stack"
    assert long_stack_cfg.upstream_alias_semantics_preserved is False
    assert type(short_stack_cfg).__name__ == "MacUR10ShortSuctionStackEnvCfg"
    assert short_stack_cfg.num_envs == 13
    assert short_stack_cfg.action_space == 7
    assert short_stack_cfg.observation_space == 40
    assert short_stack_cfg.semantic_contract == "reduced-analytic-suction-stack"
    assert short_stack_cfg.upstream_alias_semantics_preserved is False
    assert spec_2f140.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert spec_2f140.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert play_spec_2f140.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert play_spec_2f140.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ros_spec_2f140.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-ros-inference"
    assert ros_spec_2f140.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert spec_2f85.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert spec_2f85.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert play_spec_2f85.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-analytic-assembly"
    assert play_spec_2f85.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False
    assert ros_spec_2f85.kwargs[registry.TASK_CONTRACT_KEY]["semantic_contract"] == "reduced-no-ros-inference"
    assert ros_spec_2f85.kwargs[registry.TASK_CONTRACT_KEY]["upstream_alias_semantics_preserved"] is False


def test_parse_env_cfg_supports_ur10_suction_stack_task_cfgs(monkeypatch):
    """parse_env_cfg should resolve the mac-native UR10 suction stack configs without Isaac Sim imports."""

    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")
    parse_cfg = importlib.import_module("isaaclab_tasks.utils.parse_cfg")
    registry = importlib.import_module("isaaclab_tasks.registry")

    long_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0", device="cpu", num_envs=12)
    short_cfg = parse_cfg.parse_env_cfg("Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0", device="cpu", num_envs=13)
    long_spec = gym.spec("Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0")
    short_spec = gym.spec("Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0")

    assert type(long_cfg).__name__ == "MacUR10LongSuctionStackEnvCfg"
    assert long_cfg.num_envs == 12
    assert long_cfg.action_space == 7
    assert long_cfg.observation_space == 40
    assert long_cfg.semantic_contract == "reduced-analytic-suction-stack"
    assert long_cfg.upstream_alias_semantics_preserved is False
    assert type(short_cfg).__name__ == "MacUR10ShortSuctionStackEnvCfg"
    assert short_cfg.num_envs == 13
    assert short_cfg.action_space == 7
    assert short_cfg.observation_space == 40
    assert short_cfg.semantic_contract == "reduced-analytic-suction-stack"
    assert short_cfg.upstream_alias_semantics_preserved is False
    assert long_spec.id == "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0"
    assert short_spec.id == "Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0"


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
        parse_cfg.parse_env_cfg("Isaac-Repose-Cube-Shadow-Vision-Direct-v0", device="cpu")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        parse_cfg.parse_env_cfg("Isaac-Tracking-LocoManip-Digit-v0", device="cpu")
