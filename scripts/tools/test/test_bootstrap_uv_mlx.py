# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the uv bootstrap helper."""

from __future__ import annotations

from pathlib import Path

from scripts.bootstrap_uv_mlx import EditableInstall, build_install_targets, main


def test_build_install_targets_defaults_to_core_tasks_and_rl():
    args = type("Args", (), {"with_tasks": True, "with_rl": True, "dev": True})()

    installs = build_install_targets(args)

    assert installs == [
        EditableInstall(Path(__file__).resolve().parents[3] / "source" / "isaaclab", ("macos-mlx", "dev")),
        EditableInstall(Path(__file__).resolve().parents[3] / "source" / "isaaclab_tasks"),
        EditableInstall(Path(__file__).resolve().parents[3] / "source" / "isaaclab_rl", ("dev",)),
    ]


def test_build_install_targets_can_skip_optional_packages():
    args = type("Args", (), {"with_tasks": False, "with_rl": False, "dev": False})()

    installs = build_install_targets(args)

    assert installs == [
        EditableInstall(Path(__file__).resolve().parents[3] / "source" / "isaaclab", ("macos-mlx",)),
    ]


def test_main_dry_run_prints_uv_commands(monkeypatch, capsys):
    monkeypatch.setattr("scripts.bootstrap_uv_mlx.ensure_uv_available", lambda: "uv")

    exit_code = main(["--dry-run", "--venv", ".venv-test"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "uv venv" in captured.out
    assert "uv pip install" in captured.out
    assert "uv run --python" in captured.out
