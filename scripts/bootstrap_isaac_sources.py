# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bootstrap Isaac source repositories for MLX/macOS porting work."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RepoSpec:
    """Repository metadata for a source checkout."""

    name: str
    url: str
    default_ref: str
    use_lfs: bool = False


@dataclass(frozen=True)
class CloneResult:
    """Manifest entry describing a cloned source repository."""

    name: str
    path: str
    remote: str
    requested_ref: str
    resolved_sha: str
    lfs_enabled: bool


DEFAULT_REPOS = {
    "isaaclab": RepoSpec(
        name="IsaacLab",
        url="https://github.com/isaac-sim/IsaacLab.git",
        default_ref="main",
    ),
    "isaacsim": RepoSpec(
        name="IsaacSim",
        url="https://github.com/isaac-sim/IsaacSim.git",
        default_ref="main",
        use_lfs=True,
    ),
    "isaac_ros": RepoSpec(
        name="isaac_ros_common",
        url="https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git",
        default_ref="main",
    ),
}

SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the bootstrap flow."""
    parser = argparse.ArgumentParser(description="Clone Isaac source repositories for MLX/macOS port work.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory where the repositories should be cloned.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Branch, tag, or commit SHA to checkout in each cloned repository. Defaults to each repo's default ref.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Optional shallow clone depth. Ignored for commit-SHA checkouts unless a matching fetch is required.",
    )
    parser.add_argument(
        "--with-isaacsim",
        action="store_true",
        help="Also clone IsaacSim into the destination workspace.",
    )
    parser.add_argument(
        "--with-isaac-ros",
        action="store_true",
        help="Also clone isaac_ros_common into the destination workspace.",
    )
    parser.add_argument(
        "--lfs",
        action="store_true",
        help="Run `git lfs install` and `git lfs pull` for repositories that require LFS, currently IsaacSim.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for the emitted JSON manifest. Defaults to <dest>/isaac_sources_manifest.json.",
    )
    parser.add_argument(
        "--isaaclab-url",
        type=str,
        default=DEFAULT_REPOS["isaaclab"].url,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--isaacsim-url",
        type=str,
        default=DEFAULT_REPOS["isaacsim"].url,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--isaac-ros-url",
        type=str,
        default=DEFAULT_REPOS["isaac_ros"].url,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and raise with useful stderr on failure."""
    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            list(command),
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def is_commit_sha(ref: str | None) -> bool:
    """Return True when the requested ref looks like a commit SHA."""
    return bool(ref and SHA_PATTERN.fullmatch(ref))


def clone_repository(
    repo: RepoSpec,
    *,
    dest: Path,
    requested_ref: str | None = None,
    depth: int | None = None,
    enable_lfs: bool = False,
) -> CloneResult:
    """Clone a repository and resolve its checked-out commit SHA."""
    repo_path = dest / repo.name
    if repo_path.exists():
        raise FileExistsError(f"Destination already exists: {repo_path}")

    requested_ref = requested_ref or repo.default_ref
    clone_command = ["git", "clone"]
    if depth is not None and not is_commit_sha(requested_ref):
        clone_command.extend(["--depth", str(depth)])
    if requested_ref and not is_commit_sha(requested_ref):
        clone_command.extend(["--branch", requested_ref])
    clone_command.extend([repo.url, str(repo_path)])
    run_command(clone_command)

    if requested_ref and is_commit_sha(requested_ref):
        fetch_command = ["git", "-C", str(repo_path), "fetch", "origin", requested_ref]
        if depth is not None:
            fetch_command.extend(["--depth", str(depth)])
        run_command(fetch_command)
        run_command(["git", "-C", str(repo_path), "checkout", requested_ref])

    if enable_lfs and repo.use_lfs:
        run_command(["git", "lfs", "install"], cwd=repo_path)
        run_command(["git", "lfs", "pull"], cwd=repo_path)

    resolved_sha = run_command(["git", "-C", str(repo_path), "rev-parse", "HEAD"]).stdout.strip()
    return CloneResult(
        name=repo.name,
        path=str(repo_path),
        remote=repo.url,
        requested_ref=requested_ref,
        resolved_sha=resolved_sha,
        lfs_enabled=enable_lfs and repo.use_lfs,
    )


def write_manifest(results: list[CloneResult], manifest_path: Path) -> Path:
    """Write clone results to a JSON manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repositories": [asdict(result) for result in results],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def build_repo_specs(args: argparse.Namespace) -> list[RepoSpec]:
    """Build the clone set from CLI arguments."""
    repos = [
        RepoSpec(
            name=DEFAULT_REPOS["isaaclab"].name,
            url=args.isaaclab_url,
            default_ref=DEFAULT_REPOS["isaaclab"].default_ref,
            use_lfs=DEFAULT_REPOS["isaaclab"].use_lfs,
        )
    ]
    if args.with_isaacsim:
        repos.append(
            RepoSpec(
                name=DEFAULT_REPOS["isaacsim"].name,
                url=args.isaacsim_url,
                default_ref=DEFAULT_REPOS["isaacsim"].default_ref,
                use_lfs=DEFAULT_REPOS["isaacsim"].use_lfs,
            )
        )
    if args.with_isaac_ros:
        repos.append(
            RepoSpec(
                name=DEFAULT_REPOS["isaac_ros"].name,
                url=args.isaac_ros_url,
                default_ref=DEFAULT_REPOS["isaac_ros"].default_ref,
                use_lfs=DEFAULT_REPOS["isaac_ros"].use_lfs,
            )
        )
    return repos


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bootstrap flow."""
    args = parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

    results = [
        clone_repository(
            repo,
            dest=args.dest,
            requested_ref=args.ref,
            depth=args.depth,
            enable_lfs=args.lfs,
        )
        for repo in build_repo_specs(args)
    ]

    manifest_path = args.manifest or args.dest / "isaac_sources_manifest.json"
    write_manifest(results, manifest_path)

    for result in results:
        print(f"[bootstrap] cloned {result.name} @ {result.resolved_sha} -> {result.path}")
    print(f"[bootstrap] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
