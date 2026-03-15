# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
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


def test_mlx_cli_module_evaluate_writes_json_payload(tmp_path: Path):
    output_path = tmp_path / "module-eval.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "evaluate",
            "--task",
            "h1-rough",
            "--num-envs",
            "8",
            "--episodes",
            "1",
            "--episode-length-s",
            "0.5",
            "--max-steps",
            "256",
            "--json-out",
            str(output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["task"] == "h1-rough"
    assert payload["episodes_completed"] == 1


def test_mlx_cli_module_train_writes_json_payload(tmp_path: Path):
    checkpoint_path = tmp_path / "module-train-policy.npz"
    output_path = tmp_path / "module-train.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "train",
            "--task",
            "cartpole",
            "--num-envs",
            "8",
            "--updates",
            "1",
            "--rollout-steps",
            "8",
            "--epochs-per-update",
            "1",
            "--episode-length-s",
            "0.5",
            "--checkpoint",
            str(checkpoint_path),
            "--json-out",
            str(output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["task"] == "cartpole"
    assert Path(payload["checkpoint_path"]).exists()
