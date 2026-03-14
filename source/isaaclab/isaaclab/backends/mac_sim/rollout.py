# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared rollout and replay helpers for mac-native tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


def _clone_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, mx.array):
        return mx.array(value)
    return value


def _iter_arrays(value: Any) -> list[mx.array]:
    if isinstance(value, dict):
        arrays: list[mx.array] = []
        for item in value.values():
            arrays.extend(_iter_arrays(item))
        return arrays
    if isinstance(value, (list, tuple)):
        arrays = []
        for item in value:
            arrays.extend(_iter_arrays(item))
        return arrays
    if isinstance(value, mx.array):
        return [value]
    return []


@dataclass
class RolloutTrace:
    """Recorded rollout trace for deterministic replay and smoke tests."""

    initial_observations: Any
    actions: list[Any] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    rewards: list[Any] = field(default_factory=list)
    terminated: list[Any] = field(default_factory=list)
    truncated: list[Any] = field(default_factory=list)
    extras: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for diagnostics payloads."""
        final_observations = self.observations[-1] if self.observations else self.initial_observations
        reward_arrays = [reward for reward in self.rewards if isinstance(reward, mx.array)]
        terminated_arrays = [done for done in self.terminated if isinstance(done, mx.array)]
        truncated_arrays = [done for done in self.truncated if isinstance(done, mx.array)]
        reward_total = float(sum(mx.mean(reward).item() for reward in reward_arrays)) if reward_arrays else 0.0
        terminated_total = int(sum(mx.sum(done.astype(mx.int32)).item() for done in terminated_arrays)) if terminated_arrays else 0
        truncated_total = int(sum(mx.sum(done.astype(mx.int32)).item() for done in truncated_arrays)) if truncated_arrays else 0
        return {
            "steps": len(self.actions),
            "reward_total": reward_total,
            "terminated_total": terminated_total,
            "truncated_total": truncated_total,
            "final_observation_shapes": {
                key: list(value.shape) for key, value in final_observations.items()
            }
            if isinstance(final_observations, dict)
            else None,
        }


def rollout_env(
    env: Any,
    actions: Any,
    *,
    steps: int,
    policy: Callable[[Any, int], Any] | None = None,
    sync_callback: Callable[[list[mx.array]], None] | None = None,
) -> RolloutTrace:
    """Run a rollout with fixed actions, an action sequence, or a policy callback."""
    observations, _ = env.reset()
    trace = RolloutTrace(initial_observations=_clone_tree(observations))

    action_sequence: Sequence[Any] | None = actions if isinstance(actions, (list, tuple)) else None

    for step in range(steps):
        if policy is not None:
            step_action = policy(observations, step)
        elif action_sequence is not None:
            step_action = action_sequence[step]
        else:
            step_action = actions

        next_observations, rewards, terminated, truncated, extras = env.step(step_action)
        if sync_callback is not None:
            sync_callback(_iter_arrays((next_observations, rewards, terminated, truncated)))

        trace.actions.append(_clone_tree(step_action))
        trace.observations.append(_clone_tree(next_observations))
        trace.rewards.append(_clone_tree(rewards))
        trace.terminated.append(_clone_tree(terminated))
        trace.truncated.append(_clone_tree(truncated))
        trace.extras.append(dict(extras))
        observations = next_observations

    return trace


def replay_actions(
    env: Any,
    action_history: Sequence[Any],
    *,
    sync_callback: Callable[[list[mx.array]], None] | None = None,
) -> RolloutTrace:
    """Replay a previously recorded action history from a fresh reset."""
    return rollout_env(env, list(action_history), steps=len(action_history), sync_callback=sync_callback)
