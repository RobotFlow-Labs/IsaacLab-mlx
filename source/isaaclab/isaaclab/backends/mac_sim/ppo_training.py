# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared PPO helpers for the mac-native MLX training slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import mlx.core as mx


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    """Return the JSON sidecar path used for checkpoint metadata."""

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix:
        return checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.json")
    return checkpoint_path.with_suffix(".json")


def read_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata if the JSON sidecar exists."""

    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_checkpoint_metadata(checkpoint_path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Persist checkpoint metadata alongside the model weights."""

    metadata_path = checkpoint_metadata_path(checkpoint_path)
    metadata_path.write_text(json.dumps(dict(metadata), indent=2), encoding="utf-8")
    return metadata_path


def build_checkpoint_metadata(
    *,
    hidden_dim: int,
    observation_space: Any,
    action_space: Any,
    task_id: str,
    policy_distribution: str,
    train_cfg: Mapping[str, Any],
    action_std: float | None = None,
    policy_action_space: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the stable metadata payload used by the MLX task slices."""

    metadata = {
        "metadata_version": 2,
        "checkpoint_format": "isaaclab-mlx-ppo",
        "task_id": task_id,
        "policy_distribution": policy_distribution,
        "hidden_dim": hidden_dim,
        "observation_space": observation_space,
        "action_space": action_space,
        "policy_action_space": action_space if policy_action_space is None else policy_action_space,
        "train_cfg": dict(train_cfg),
    }
    if action_std is not None:
        metadata["action_std"] = action_std
    if extra:
        metadata.update(extra)
    return metadata


def resolve_resume_hidden_dim(resume_from: str | None, default_hidden_dim: int) -> int:
    """Recover the hidden dimension from checkpoint metadata when resuming."""

    if resume_from is None:
        return default_hidden_dim
    metadata = read_checkpoint_metadata(resume_from)
    return int(metadata.get("hidden_dim", default_hidden_dim))


def mean_recent_return(completed_returns: Sequence[float], *, window: int = 10) -> float:
    """Compute the trailing mean return used in training summaries."""

    recent = completed_returns[-window:]
    return sum(recent) / max(1, len(recent))


def normalize_advantages(advantages: mx.array, *, epsilon: float = 1e-8) -> mx.array:
    """Normalize flattened PPO advantages."""

    adv_mean = mx.mean(advantages)
    adv_std = mx.sqrt(mx.mean(mx.square(advantages - adv_mean)) + epsilon)
    return (advantages - adv_mean) / adv_std


def compute_gae(
    rewards_t: mx.array,
    dones_t: mx.array,
    values_t: mx.array,
    next_values_t: mx.array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[mx.array, mx.array]:
    """Compute GAE advantages and returns for a batched PPO rollout."""

    advantages = []
    gae = mx.zeros((values_t.shape[1],), dtype=mx.float32)

    for step in reversed(range(rewards_t.shape[0])):
        mask = 1.0 - dones_t[step]
        delta = rewards_t[step] + gamma * next_values_t[step] * mask - values_t[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.append(gae)

    advantages_t = mx.stack(list(reversed(advantages)))
    returns_t = advantages_t + values_t
    return advantages_t, returns_t


def save_policy_checkpoint(model: Any, checkpoint_path: str | Path, metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Write model weights and the JSON metadata sidecar."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path))
    metadata_path = write_checkpoint_metadata(checkpoint_path, metadata)
    return str(checkpoint_path), str(metadata_path)
