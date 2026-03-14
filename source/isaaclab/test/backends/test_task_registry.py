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
    "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    "Isaac-Cartpole-Direct-v0",
    "Isaac-Cart-Double-Pendulum-Direct-v0",
    "Isaac-Quadcopter-Direct-v0",
)
UNSUPPORTED_MAC_TASK_IDS = (
    "Isaac-Franka-Cabinet-Direct-v0",
    "Isaac-Factory-PegInsert-Direct-v0",
    "Isaac-PickPlace-GR1T2-Abs-v0",
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    "Isaac-Deploy-Reach-UR10e-v0",
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

    for task_id in SAFE_TASK_IDS:
        assert gym.spec(task_id).id == task_id

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
        parse_cfg.parse_env_cfg("Isaac-Franka-Cabinet-Direct-v0", device="cpu")

    with pytest.raises(UnsupportedBackendError, match="sim-backend=isaacsim"):
        parse_cfg.parse_env_cfg("Isaac-Repose-Cube-Shadow-Vision-Direct-v0", device="cpu")
