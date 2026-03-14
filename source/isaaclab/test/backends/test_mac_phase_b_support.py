# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase B substrate tests for mac-native terrain, contacts, determinism, and rollout helpers."""

from __future__ import annotations

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    BatchedContactSensorState,
    DeterministicResetSampler,
    MacCartpoleEnv,
    MacCartpoleEnvCfg,
    MacPlaneTerrain,
    action_rate_l2,
    base_contact_termination,
    feet_air_time_reward,
    flat_orientation_l2,
    mac_env_diagnostics,
    replay_actions,
    rollout_env,
    terrain_out_of_bounds,
    track_linear_velocity_xy_exp,
    track_yaw_rate_z_exp,
    undesired_contacts,
)


def test_reset_sampler_forks_are_stable_and_independent():
    """Reset sampler forks should produce stable streams for deterministic mac resets."""
    sampler_a = DeterministicResetSampler(17)
    sampler_b = DeterministicResetSampler(17)

    fork_a = sampler_a.fork("terrain")
    fork_b = sampler_b.fork("terrain")
    other = sampler_a.fork("goals")

    assert mx.allclose(fork_a.uniform((4,), -1.0, 1.0), fork_b.uniform((4,), -1.0, 1.0)).item()
    assert not mx.allclose(fork_a.uniform((4,), -1.0, 1.0), other.uniform((4,), -1.0, 1.0)).item()


def test_plane_terrain_and_contact_model_cover_flat_locomotion_semantics():
    """Flat terrain and contact buffers should expose the minimum quadruped locomotion substrate."""
    terrain = MacPlaneTerrain(num_envs=2, env_spacing=4.0, tile_size=(4.0, 4.0))
    contacts = BatchedContactSensorState(
        num_envs=2,
        body_names=("base", "LF_FOOT", "RF_FOOT", "LF_THIGH"),
        foot_body_names=("LF_FOOT", "RF_FOOT"),
        step_dt=0.05,
    )

    body_pos_w = mx.array(
        [
            [[0.0, 0.0, 0.40], [0.2, 0.2, 0.005], [-0.2, 0.2, 0.010], [0.1, 0.0, 0.25]],
            [[0.0, 0.0, 0.015], [0.2, 0.2, 0.080], [-0.2, 0.2, 0.004], [0.1, 0.0, 0.010]],
        ],
        dtype=mx.float32,
    )
    body_vel_w = mx.zeros_like(body_pos_w)
    body_vel_w[:, :, 2] = mx.array([[0.0, -0.2, -0.1, 0.0], [-0.3, 0.0, -0.2, -0.3]], dtype=mx.float32)

    contact_mask = contacts.update(body_pos_w, body_vel_w, terrain)

    assert contact_mask.shape == (2, 4)
    assert bool(contact_mask[0, 1].item()) is True
    assert bool(contact_mask[1, 0].item()) is True
    assert bool(contacts.compute_first_contact()[0, 0].item()) is True
    assert bool(contacts.compute_first_contact()[1, 1].item()) is True
    assert contacts.state_dict()["hotpath_backend"] == "mlx-compiled"
    assert terrain.surface_normals(body_pos_w.reshape((-1, 3))).shape == (8, 3)


def test_locomotion_reward_and_termination_utilities_use_contact_buffers():
    """Reward and termination helpers should use terrain/contact substrate directly."""
    terrain = MacPlaneTerrain(num_envs=2, env_spacing=4.0, tile_size=(4.0, 4.0))
    contacts = BatchedContactSensorState(
        num_envs=2,
        body_names=("base", "LF_FOOT", "RF_FOOT", "LF_THIGH"),
        foot_body_names=("LF_FOOT", "RF_FOOT"),
        step_dt=0.05,
    )

    airborne_pos = mx.array(
        [
            [[0.0, 0.0, 0.35], [0.2, 0.2, 0.12], [-0.2, 0.2, 0.12], [0.1, 0.0, 0.22]],
            [[4.5, 0.0, 0.35], [4.7, 0.2, 0.12], [4.3, 0.2, 0.12], [4.6, 0.0, 0.22]],
        ],
        dtype=mx.float32,
    )
    contacts.update(airborne_pos, mx.zeros_like(airborne_pos), terrain)

    landing_pos = mx.array(
        [
            [[0.0, 0.0, 0.35], [0.2, 0.2, 0.005], [-0.2, 0.2, 0.005], [0.1, 0.0, 0.010]],
            [[4.5, 0.0, 0.010], [4.7, 0.2, 0.005], [4.3, 0.2, 0.12], [4.6, 0.0, 0.005]],
        ],
        dtype=mx.float32,
    )
    landing_vel = mx.zeros_like(landing_pos)
    landing_vel[:, :, 2] = -0.2
    contacts.update(landing_pos, landing_vel, terrain)

    commands = mx.array([[0.6, 0.0, 0.1], [0.2, 0.0, -0.3]], dtype=mx.float32)
    root_lin_vel_b = mx.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.1]], dtype=mx.float32)
    root_ang_vel_b = mx.array([[0.0, 0.0, 0.12], [0.1, 0.1, -0.1]], dtype=mx.float32)
    projected_gravity_b = mx.array([[0.0, 0.0, -1.0], [0.3, 0.2, -0.9]], dtype=mx.float32)
    actions = mx.ones((2, 12), dtype=mx.float32)
    previous_actions = mx.zeros((2, 12), dtype=mx.float32)

    air_time = feet_air_time_reward(contacts, commands)
    undesired = undesired_contacts(contacts, ("LF_THIGH",), threshold=1.0)
    base_done = base_contact_termination(contacts, ("base",), threshold=1.0)
    out_of_bounds = terrain_out_of_bounds(terrain, landing_pos[:, 0, :], buffer=0.1)

    assert track_linear_velocity_xy_exp(commands, root_lin_vel_b).shape == (2,)
    assert track_yaw_rate_z_exp(commands, root_ang_vel_b).shape == (2,)
    assert action_rate_l2(actions, previous_actions).shape == (2,)
    assert flat_orientation_l2(projected_gravity_b).shape == (2,)
    assert air_time.shape == (2,)
    assert float(air_time[0].item()) <= 0.0
    assert float(undesired[0].item()) == 1.0
    assert bool(base_done[1].item()) is True
    assert bool(out_of_bounds[1].item()) is True


def test_rollout_and_replay_helpers_are_deterministic_for_cartpole():
    """Recorded mac rollouts should replay deterministically with the same seed/config."""
    cfg = MacCartpoleEnvCfg(num_envs=4, seed=23, episode_length_s=0.3)
    actions = mx.zeros((cfg.num_envs, 1), dtype=mx.float32)

    env_a = MacCartpoleEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=8)

    env_b = MacCartpoleEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()


def test_mac_env_diagnostics_report_determinism_and_rollout_support():
    """Concrete mac env diagnostics should expose subsystem support for benchmarks."""
    env = MacCartpoleEnv(MacCartpoleEnvCfg(num_envs=4, seed=31, episode_length_s=0.3))
    diagnostics = mac_env_diagnostics(env)

    assert diagnostics["sim_backend"]["subsystems"]["deterministic_resets"] is True
    assert diagnostics["supports_rollout_helpers"] is True
    assert diagnostics["determinism"]["env"]["seed"] == 31
    assert diagnostics["determinism"]["sim"]["seed"] != 31
