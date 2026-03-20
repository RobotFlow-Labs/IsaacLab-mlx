# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the maintained MLX kernel inventory."""

from __future__ import annotations

from pathlib import Path

from isaaclab.backends.kernel_inventory import CURRENT_MAC_NATIVE_TASKS, KERNEL_PORT_INVENTORY


def test_kernel_inventory_lists_expected_current_mac_native_tasks():
    """The kernel inventory should publish the current mac-native task suite as one stable tuple."""
    assert CURRENT_MAC_NATIVE_TASKS == (
        "cartpole",
        "cart-double-pendulum",
        "quadcopter",
        "anymal-c-flat",
        "anymal-c-rough",
        "h1-flat",
        "h1-rough",
        "franka-reach",
        "openarm-reach",
        "openarm-bi-reach",
        "ur10-reach",
        "ur10e-deploy-reach",
        "ur10e-gear-assembly-2f140",
        "ur10e-gear-assembly-2f85",
        "ur10-long-suction-stack",
        "ur10-short-suction-stack",
        "franka-lift",
        "openarm-lift",
        "franka-teddy-bear-lift",
        "franka-stack-instance-randomize",
        "franka-stack",
        "franka-stack-rgb",
        "franka-bin-stack",
        "franka-cabinet",
        "franka-open-drawer",
        "openarm-open-drawer",
    )


def test_kernel_inventory_items_reference_real_upstream_modules():
    """Every inventory entry should point at tracked source files in the repo."""
    repo_root = Path(__file__).resolve().parents[4]

    assert {item.key for item in KERNEL_PORT_INVENTORY} == {
        "raycast-mesh-ops",
        "wrench-composer",
        "fabric-transform-kernels",
        "tiled-camera-reshape",
    }

    for item in KERNEL_PORT_INVENTORY:
        assert item.status
        assert item.target_tasks
        assert item.symbols
        assert item.replacement_strategy
        assert item.notes
        for module_path in item.upstream_modules:
            assert (repo_root / module_path).exists(), f"Missing inventory path: {module_path}"

    assert any("manipulation-stack" in item.target_tasks for item in KERNEL_PORT_INVENTORY)
    assert any("manipulation-cabinet" in item.target_tasks for item in KERNEL_PORT_INVENTORY)
