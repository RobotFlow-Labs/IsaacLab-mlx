# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-only helpers for backend validation."""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def _probe_mlx_runtime() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-c",
        "import mlx.core as mx; value = mx.array([1.0], dtype=mx.float32); mx.eval(value); print('mlx-ok')",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0 and "mlx-ok" in output, output


def require_mlx_runtime():
    """Import MLX only after a subprocess probe confirms the runtime is usable."""
    ok, output = _probe_mlx_runtime()
    if not ok:
        pytest.skip(f"Working MLX runtime unavailable: {output or 'probe failed'}")

    import mlx.core as mx

    return mx
