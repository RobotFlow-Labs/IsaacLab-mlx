# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostics helpers for concrete mac-native simulator adapters."""

from __future__ import annotations

from typing import Any


def mac_env_diagnostics(env: Any, *, rollout_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect adapter, terrain, contact, and determinism diagnostics for a mac-native env."""
    sim_backend = env.sim_backend
    payload = {
        "sim_backend": sim_backend.state_dict(),
        "supports_rollout_helpers": True,
    }

    terrain = getattr(sim_backend, "terrain", None)
    if terrain is not None and hasattr(terrain, "state_dict"):
        payload["terrain"] = terrain.state_dict()

    contact_model = getattr(sim_backend, "contact_model", None)
    if contact_model is not None and hasattr(contact_model, "state_dict"):
        payload["contacts"] = contact_model.state_dict()

    height_scan_sensor = getattr(env, "height_scan_sensor", None)
    if height_scan_sensor is not None and hasattr(height_scan_sensor, "state_dict"):
        payload["sensor"] = height_scan_sensor.state_dict()

    reset_streams: dict[str, Any] = {}
    reset_sampler = getattr(env, "reset_sampler", None)
    if reset_sampler is not None and hasattr(reset_sampler, "state_dict"):
        reset_streams["env"] = reset_sampler.state_dict()
    sim_reset_sampler = getattr(sim_backend, "reset_sampler", None)
    if sim_reset_sampler is not None and hasattr(sim_reset_sampler, "state_dict"):
        reset_streams["sim"] = sim_reset_sampler.state_dict()
    goal_sampler = getattr(env, "goal_sampler", None)
    if goal_sampler is not None and hasattr(goal_sampler, "state_dict"):
        reset_streams["goals"] = goal_sampler.state_dict()
    if reset_streams:
        payload["determinism"] = reset_streams

    if rollout_summary is not None:
        payload["rollout"] = rollout_summary

    return payload
