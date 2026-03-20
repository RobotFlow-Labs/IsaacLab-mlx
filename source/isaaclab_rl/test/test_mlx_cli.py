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


def test_mlx_cli_module_normalizes_upstream_manipulation_aliases(tmp_path: Path):
    eval_output_path = tmp_path / "module-alias-eval.json"
    train_output_path = tmp_path / "module-alias-train.json"
    checkpoint_path = tmp_path / "module-alias-train-policy.npz"

    eval_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "evaluate",
            "--task",
            "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0",
            "--num-envs",
            "8",
            "--episodes",
            "1",
            "--episode-length-s",
            "0.5",
            "--max-steps",
            "256",
            "--json-out",
            str(eval_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert eval_result.returncode == 0, eval_result.stderr
    eval_payload = json.loads(eval_output_path.read_text(encoding="utf-8"))
    assert eval_payload["task"] == "franka-stack-rgb"
    assert eval_payload["episodes_completed"] == 1

    instance_eval_output_path = tmp_path / "module-instance-alias-eval.json"
    instance_eval_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "evaluate",
            "--task",
            "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
            "--num-envs",
            "8",
            "--episodes",
            "1",
            "--episode-length-s",
            "0.5",
            "--max-steps",
            "256",
            "--json-out",
            str(instance_eval_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert instance_eval_result.returncode == 0, instance_eval_result.stderr
    instance_eval_payload = json.loads(instance_eval_output_path.read_text(encoding="utf-8"))
    assert instance_eval_payload["task"] == "franka-stack-instance-randomize"
    assert instance_eval_payload["episodes_completed"] == 1

    bin_eval_output_path = tmp_path / "module-bin-alias-eval.json"
    bin_eval_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "evaluate",
            "--task",
            "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
            "--num-envs",
            "8",
            "--episodes",
            "1",
            "--episode-length-s",
            "0.5",
            "--max-steps",
            "256",
            "--json-out",
            str(bin_eval_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert bin_eval_result.returncode == 0, bin_eval_result.stderr
    bin_eval_payload = json.loads(bin_eval_output_path.read_text(encoding="utf-8"))
    assert bin_eval_payload["task"] == "franka-bin-stack"
    assert bin_eval_payload["episodes_completed"] == 1
    assert bin_eval_payload["task_spec"]["semantic_contract"] == "reduced-no-mimic"
    assert bin_eval_payload["task_spec"]["upstream_alias_semantics_preserved"] is False

    bin_train_output_path = tmp_path / "module-bin-alias-train.json"
    bin_checkpoint_path = tmp_path / "module-bin-alias-train-policy.npz"
    bin_train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "train",
            "--task",
            "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
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
            str(bin_checkpoint_path),
            "--json-out",
            str(bin_train_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert bin_train_result.returncode == 0, bin_train_result.stderr
    bin_train_payload = json.loads(bin_train_output_path.read_text(encoding="utf-8"))
    assert bin_train_payload["task"] == "franka-bin-stack"
    assert Path(bin_train_payload["checkpoint_path"]).exists()
    assert bin_train_payload["task_spec"]["semantic_contract"] == "reduced-no-mimic"
    assert bin_train_payload["task_spec"]["upstream_alias_semantics_preserved"] is False

    ur10e_eval_output_path = tmp_path / "module-ur10e-alias-eval.json"
    ur10e_eval_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "evaluate",
            "--task",
            "Isaac-Deploy-Reach-UR10e-Play-v0",
            "--num-envs",
            "8",
            "--episodes",
            "1",
            "--episode-length-s",
            "0.5",
            "--max-steps",
            "256",
            "--json-out",
            str(ur10e_eval_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert ur10e_eval_result.returncode == 0, ur10e_eval_result.stderr
    ur10e_eval_payload = json.loads(ur10e_eval_output_path.read_text(encoding="utf-8"))
    assert ur10e_eval_payload["task"] == "ur10e-deploy-reach"
    assert ur10e_eval_payload["episodes_completed"] == 1
    assert ur10e_eval_payload["task_spec"]["semantic_contract"] == "reduced-analytic-pose"
    assert ur10e_eval_payload["task_spec"]["upstream_alias_semantics_preserved"] is False

    ur10e_train_output_path = tmp_path / "module-ur10e-alias-train.json"
    ur10e_checkpoint_path = tmp_path / "module-ur10e-alias-train-policy.npz"
    ur10e_train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "train",
            "--task",
            "Isaac-Deploy-Reach-UR10e-v0",
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
            str(ur10e_checkpoint_path),
            "--json-out",
            str(ur10e_train_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert ur10e_train_result.returncode == 0, ur10e_train_result.stderr
    ur10e_train_payload = json.loads(ur10e_train_output_path.read_text(encoding="utf-8"))
    assert ur10e_train_payload["task"] == "ur10e-deploy-reach"
    assert Path(ur10e_train_payload["checkpoint_path"]).exists()
    assert ur10e_train_payload["task_spec"]["semantic_contract"] == "reduced-analytic-pose"
    assert ur10e_train_payload["task_spec"]["upstream_alias_semantics_preserved"] is False

    train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "train",
            "--task",
            "Isaac-Open-Drawer-Franka-IK-Rel-v0",
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
            str(train_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert train_result.returncode == 0, train_result.stderr
    train_payload = json.loads(train_output_path.read_text(encoding="utf-8"))
    assert train_payload["task"] == "franka-open-drawer"
    assert Path(train_payload["checkpoint_path"]).exists()


def test_mlx_cli_module_handles_lift_alias_and_rejects_unsupported_manipulation_id(tmp_path: Path):
    train_output_path = tmp_path / "module-lift-alias-train.json"
    checkpoint_path = tmp_path / "module-lift-alias-policy.npz"

    train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "train",
            "--task",
            "Isaac-Lift-Cube-Franka-IK-Rel-v0",
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
            str(train_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert train_result.returncode == 0, train_result.stderr
    train_payload = json.loads(train_output_path.read_text(encoding="utf-8"))
    assert train_payload["task"] == "franka-lift"
    assert Path(train_payload["checkpoint_path"]).exists()

    unsupported_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "evaluate",
            "--task",
            "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert unsupported_result.returncode != 0
    assert "invalid choice" in unsupported_result.stderr


def test_mlx_cli_module_handles_teddy_bear_lift_alias(tmp_path: Path):
    train_output_path = tmp_path / "module-teddy-bear-lift-train.json"
    checkpoint_path = tmp_path / "module-teddy-bear-lift-policy.npz"

    train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_rl.mlx_cli",
            "train",
            "--task",
            "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
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
            str(train_output_path),
        ],
        cwd=_repo_root(),
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert train_result.returncode == 0, train_result.stderr
    train_payload = json.loads(train_output_path.read_text(encoding="utf-8"))
    assert train_payload["task"] == "franka-teddy-bear-lift"
    assert Path(train_payload["checkpoint_path"]).exists()


def test_mlx_cli_module_normalizes_reduced_openarm_and_ur10_aliases(tmp_path: Path):
    eval_cases = [
        ("Isaac-Reach-OpenArm-Play-v0", "openarm-reach"),
        ("Isaac-Reach-OpenArm-Bi-Play-v0", "openarm-bi-reach"),
        ("Isaac-Reach-UR10-Play-v0", "ur10-reach"),
        ("Isaac-Open-Drawer-OpenArm-Play-v0", "openarm-open-drawer"),
    ]
    train_cases = [
        ("Isaac-Reach-OpenArm-v0", "openarm-reach"),
        ("Isaac-Reach-OpenArm-Bi-v0", "openarm-bi-reach"),
        ("Isaac-Reach-UR10-v0", "ur10-reach"),
        ("Isaac-Lift-Cube-OpenArm-v0", "openarm-lift"),
        ("Isaac-Open-Drawer-OpenArm-v0", "openarm-open-drawer"),
    ]

    for alias, canonical in eval_cases:
        output_path = tmp_path / f"{canonical}-eval.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "isaaclab_rl.mlx_cli",
                "evaluate",
                "--task",
                alias,
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
        assert payload["task"] == canonical
        assert payload["episodes_completed"] == 1

    for index, (alias, canonical) in enumerate(train_cases, start=1):
        output_path = tmp_path / f"{canonical}-train.json"
        checkpoint_path = tmp_path / f"{canonical}-policy-{index}.npz"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "isaaclab_rl.mlx_cli",
                "train",
                "--task",
                alias,
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
        assert payload["task"] == canonical
        assert Path(payload["checkpoint_path"]).exists()
