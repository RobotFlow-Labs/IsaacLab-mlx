# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight Nucleus path constants that can be imported without Isaac Sim."""

from __future__ import annotations

import os

NUCLEUS_ASSET_ROOT_ENV_VAR = "ISAACLAB_NUCLEUS_ASSET_ROOT"
"""Environment variable that can override the root Nucleus asset path."""


def _resolve_nucleus_asset_root() -> str:
    """Resolve the root Nucleus asset directory without requiring Isaac Sim modules."""
    override = os.environ.get(NUCLEUS_ASSET_ROOT_ENV_VAR)
    if override:
        return override.rstrip("/")

    try:
        import carb
    except Exception:
        return ""

    value = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    if not value:
        return ""
    return str(value).rstrip("/")


def _join_nucleus_path(root: str, *parts: str) -> str:
    """Join a root path with path segments while preserving import safety."""
    clean_parts = [part.strip("/") for part in parts if part]
    if not root:
        return "/".join(clean_parts)
    return "/".join([root.rstrip("/"), *clean_parts])


NUCLEUS_ASSET_ROOT_DIR = _resolve_nucleus_asset_root()
"""Path to the root directory on the Nucleus Server, if available."""

NVIDIA_NUCLEUS_DIR = _join_nucleus_path(NUCLEUS_ASSET_ROOT_DIR, "NVIDIA")
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR = _join_nucleus_path(NUCLEUS_ASSET_ROOT_DIR, "Isaac")
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = _join_nucleus_path(ISAAC_NUCLEUS_DIR, "IsaacLab")
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""
