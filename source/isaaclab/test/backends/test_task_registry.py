# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLX/mac task registry surface."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import gymnasium as gym

from isaaclab.backends import resolve_runtime_selection, set_runtime_selection

SAFE_TASK_IDS = (
    "Isaac-Cartpole-Direct-v0",
    "Isaac-Cart-Double-Pendulum-Direct-v0",
    "Isaac-Quadcopter-Direct-v0",
)


def _clear_task_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "isaaclab_tasks" or module_name.startswith("isaaclab_tasks."):
            del sys.modules[module_name]


def _clear_task_specs() -> None:
    for task_id in SAFE_TASK_IDS + ("Isaac-Franka-Cabinet-Direct-v0",):
        gym.registry.pop(task_id, None)


def test_mlx_task_registry_registers_supported_mac_tasks(monkeypatch):
    """The mac runtime should register the MLX-supported task ids without importing the full task tree."""
    task_source = Path(__file__).resolve().parents[3] / "isaaclab_tasks"
    monkeypatch.syspath_prepend(str(task_source))
    _clear_task_modules()
    _clear_task_specs()
    set_runtime_selection(resolve_runtime_selection(compute_backend="mlx", sim_backend="mac-sim", device="cpu"))

    importlib.import_module("isaaclab_tasks")

    for task_id in SAFE_TASK_IDS:
        assert gym.spec(task_id).id == task_id

    try:
        gym.spec("Isaac-Franka-Cabinet-Direct-v0")
    except gym.error.Error:
        pass
    else:
        raise AssertionError("Unsupported Isaac Sim tasks should not be registered on the mac bootstrap path.")


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
