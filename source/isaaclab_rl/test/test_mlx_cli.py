# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:source/isaaclab:source/isaaclab_rl:{env.get('PYTHONPATH', '')}".rstrip(":")
    return env


def test_mlx_cli_module_exposes_train_help():
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab_rl.mlx_cli", "train", "--help"],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--task" in result.stdout


def test_mlx_cli_module_exposes_evaluate_help():
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab_rl.mlx_cli", "evaluate", "--help"],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--task" in result.stdout
