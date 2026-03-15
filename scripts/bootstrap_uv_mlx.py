# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bootstrap a uv-managed MLX/mac development environment for this fork."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class EditableInstall:
    """One editable package install target plus optional extras."""

    path: Path
    extras: tuple[str, ...] = ()

    def requirement(self) -> str:
        if not self.extras:
            return str(self.path)
        return f"{self.path}[{','.join(self.extras)}]"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the MLX/mac IsaacLab fork with uv.")
    parser.add_argument("--venv", type=Path, default=Path(".venv"), help="Virtualenv path to create/update.")
    parser.add_argument("--python", dest="python_spec", default=None, help="Optional Python version or interpreter path for `uv venv`.")
    parser.add_argument(
        "--with-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install the lazy mac task registry package.",
    )
    parser.add_argument(
        "--with-rl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install the public MLX wrapper package.",
    )
    parser.add_argument(
        "--dev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include dev extras for editable source packages.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the uv commands without executing them.")
    return parser.parse_args(argv)


def ensure_uv_available() -> str:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise SystemExit("`uv` is not installed or not on PATH.")
    return uv_bin


def python_executable_for_venv(venv_path: Path) -> Path:
    if shutil.which("python.exe") and (venv_path / "Scripts" / "python.exe").exists():
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def build_install_targets(args: argparse.Namespace) -> list[EditableInstall]:
    root = repo_root()
    core_extras = ["macos-mlx"]
    rl_extras: list[str] = []
    if args.dev:
        core_extras.append("dev")
        rl_extras.append("dev")

    installs = [EditableInstall(root / "source" / "isaaclab", tuple(core_extras))]
    if args.with_tasks:
        installs.append(EditableInstall(root / "source" / "isaaclab_tasks"))
    if args.with_rl:
        installs.append(EditableInstall(root / "source" / "isaaclab_rl", tuple(rl_extras)))
    return installs


def run_command(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def bootstrap(args: argparse.Namespace) -> int:
    uv_bin = ensure_uv_available()
    root = repo_root()
    venv_path = (root / args.venv).resolve() if not args.venv.is_absolute() else args.venv.resolve()

    venv_command = [uv_bin, "venv", str(venv_path)]
    if args.python_spec:
        venv_command.extend(["--python", args.python_spec])
    run_command(venv_command, cwd=root, dry_run=args.dry_run)

    python_executable = python_executable_for_venv(venv_path)
    install_command = [uv_bin, "pip", "install", "--python", str(python_executable)]
    for target in build_install_targets(args):
        install_command.extend(["-e", target.requirement()])
    run_command(install_command, cwd=root, dry_run=args.dry_run)

    print("")
    print("Next commands:")
    print(f"  uv run --python {python_executable} scripts/reinforcement_learning/mlx/evaluate_task.py --task cartpole")
    print(f"  uv run --python {python_executable} scripts/tools/mac_planner_smoke.py logs/planner/mac-planner-smoke.json")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return bootstrap(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
