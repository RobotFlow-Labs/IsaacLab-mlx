# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the source bootstrap script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.bootstrap_isaac_sources import CloneResult, RepoSpec, build_repo_specs, main, write_manifest


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _create_local_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "--initial-branch=main"], cwd=path)
    _run(["git", "config", "user.name", "Codex"], cwd=path)
    _run(["git", "config", "user.email", "codex@example.com"], cwd=path)
    (path / "README.md").write_text("# test\n")
    _run(["git", "add", "README.md"], cwd=path)
    _run(["git", "commit", "-m", "initial"], cwd=path)
    return _run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()


def test_main_clones_isaaclab_and_writes_manifest(tmp_path: Path):
    """Clone a local IsaacLab mirror and record its resolved SHA."""
    source_repo = tmp_path / "source-isaaclab"
    expected_sha = _create_local_repo(source_repo)
    workspace = tmp_path / "workspace"

    exit_code = main(
        [
            "--dest",
            str(workspace),
            "--isaaclab-url",
            str(source_repo),
        ]
    )

    assert exit_code == 0
    clone_path = workspace / "IsaacLab"
    manifest_path = workspace / "isaac_sources_manifest.json"
    assert clone_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["repositories"] == [
        {
            "name": "IsaacLab",
            "path": str(clone_path),
            "remote": str(source_repo),
            "requested_ref": "main",
            "resolved_sha": expected_sha,
            "lfs_enabled": False,
        }
    ]


def test_build_repo_specs_includes_optional_repositories():
    """Optional clone switches should add IsaacSim and isaac_ros_common."""
    args = type(
        "Args",
        (),
        {
            "with_isaacsim": True,
            "with_isaac_ros": True,
            "isaaclab_url": "https://example.com/isaaclab.git",
            "isaacsim_url": "https://example.com/isaacsim.git",
            "isaac_ros_url": "https://example.com/isaac_ros_common.git",
        },
    )()

    repos = build_repo_specs(args)

    assert [repo.name for repo in repos] == ["IsaacLab", "IsaacSim", "isaac_ros_common"]
    assert repos[1].use_lfs is True


def test_write_manifest_serializes_clone_results(tmp_path: Path):
    """Manifest writing should preserve all clone metadata."""
    manifest_path = write_manifest(
        [
            CloneResult(
                name="IsaacLab",
                path=str(tmp_path / "IsaacLab"),
                remote="https://github.com/isaac-sim/IsaacLab.git",
                requested_ref="main",
                resolved_sha="abc1234",
                lfs_enabled=False,
            )
        ],
        tmp_path / "manifest.json",
    )

    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["repositories"][0]["resolved_sha"] == "abc1234"


def test_clone_repository_lfs_is_only_enabled_for_lfs_repositories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The optional LFS flow should run only for repos that opt into it."""
    from scripts import bootstrap_isaac_sources as bootstrap

    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command: list[str], *, cwd: Path | None = None):
        calls.append((command, None if cwd is None else str(cwd)))
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[-3:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "run_command", fake_run)

    bootstrap.clone_repository(
        RepoSpec(name="IsaacSim", url="https://example.com/IsaacSim.git", default_ref="main", use_lfs=True),
        dest=tmp_path,
        enable_lfs=True,
    )

    assert (["git", "lfs", "install"], str(tmp_path / "IsaacSim")) in calls
    assert (["git", "lfs", "pull"], str(tmp_path / "IsaacSim")) in calls

