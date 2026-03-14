# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic reset sampling helpers for mac-native simulator tasks."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import numpy as np


def _tag_to_int(tag: str | int) -> int:
    """Convert a stable fork tag into a deterministic integer."""
    if isinstance(tag, int):
        return tag
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


class DeterministicResetSampler:
    """Small deterministic RNG wrapper used by mac-native env reset paths."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.draw_count = 0

    def fork(self, tag: str | int) -> "DeterministicResetSampler":
        """Derive a child sampler with a stable independent seed."""
        return DeterministicResetSampler(self.seed ^ _tag_to_int(tag))

    def uniform(self, shape: tuple[int, ...] | list[int] | int, low: float, high: float, *, dtype=mx.float32) -> mx.array:
        """Sample a uniform MLX array with deterministic ordering."""
        values = self._rng.uniform(low=low, high=high, size=shape)
        self.draw_count += int(np.prod(np.shape(values)))
        return mx.array(values, dtype=dtype)

    def integers(self, shape: tuple[int, ...] | list[int] | int, low: int, high: int, *, dtype=mx.int32) -> mx.array:
        """Sample integer values with deterministic ordering."""
        values = self._rng.integers(low=low, high=high, size=shape)
        self.draw_count += int(np.prod(np.shape(values)))
        return mx.array(values, dtype=dtype)

    def stagger_episode_lengths(self, num_envs: int, max_episode_length: int) -> mx.array:
        """Generate deterministic staggered episode offsets for vectorized resets."""
        if max_episode_length <= 0:
            return mx.zeros((num_envs,), dtype=mx.int32)
        return self.integers((num_envs,), 0, max_episode_length, dtype=mx.int32)

    def state_dict(self) -> dict[str, Any]:
        """Return a compact diagnostics payload."""
        return {
            "seed": self.seed,
            "draw_count": self.draw_count,
            "bit_generator": type(self._rng.bit_generator).__name__,
        }
