# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portability tests for utility modules used by the MLX/mac path."""

from __future__ import annotations

import os
import subprocess
import sys
import importlib

import numpy as np
import pytest

from isaaclab.utils.seed import configure_seed

array_utils = importlib.import_module("isaaclab.utils.array")


def test_convert_to_torch_handles_missing_torch_dependency():
    """convert_to_torch should fail explicitly when PyTorch is unavailable."""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    if array_utils.torch is None:
        with pytest.raises(ModuleNotFoundError, match="PyTorch is required"):
            array_utils.convert_to_torch(array)
    else:
        tensor = array_utils.convert_to_torch(array)
        assert tensor.shape == (3,)


def test_configure_seed_runs_without_cuda():
    """Seed utility should work even when no CUDA device is present."""
    value = configure_seed(123, torch_deterministic=False)
    assert value == 123


def test_utils_import_without_warp_module():
    """Importing core utility modules should not fail when Warp is unavailable."""
    code = """
import importlib
import sys

sys.modules['warp'] = None
array_mod = importlib.import_module('isaaclab.utils.array')
seed_mod = importlib.import_module('isaaclab.utils.seed')
assert 'warp' not in array_mod.TENSOR_TYPES
assert hasattr(seed_mod, 'configure_seed')
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_seed_module_import_without_torch_or_warp():
    """Seed utilities should still import when torch/warp are both unavailable."""
    code = """
import importlib
import sys

sys.modules['torch'] = None
sys.modules['warp'] = None
seed_mod = importlib.import_module('isaaclab.utils.seed')
value = seed_mod.configure_seed(7, torch_deterministic=False)
assert value == 7
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_utils_dynamic_export_skips_optional_dependency_modules():
    """isaaclab.utils dynamic exports should not crash when optional deps are absent."""
    code = """
import importlib
import sys

import numpy as np

sys.modules['torch'] = None
sys.modules['warp'] = None
utils_mod = importlib.import_module('isaaclab.utils')
convert_to_torch = getattr(utils_mod, 'convert_to_torch')
try:
    convert_to_torch(np.array([1.0], dtype=np.float32))
except ModuleNotFoundError:
    print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
