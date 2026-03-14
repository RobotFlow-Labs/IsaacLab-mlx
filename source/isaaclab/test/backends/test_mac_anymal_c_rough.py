# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native rough ANYmal-C slice."""

from __future__ import annotations

from isaaclab.backends.test_utils import require_mlx_runtime

mx = require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    MacAnymalCRoughEnv,
    MacAnymalCRoughEnvCfg,
    replay_actions,
    rollout_env,
)


def test_mac_anymal_c_rough_reset_and_height_scan_shape():
    """The rough ANYmal-C env should enable the raycast-backed height scan by default."""

    cfg = MacAnymalCRoughEnvCfg(num_envs=6, seed=17, episode_length_s=0.5)
    env = MacAnymalCRoughEnv(cfg)

    obs, extras = env.reset()

    assert extras == {}
    assert env.height_scan_sensor is not None
    assert env.height_scan_dim == 9
    assert env.sim_backend.terrain.state_dict()["type"] == "wave"
    assert obs["policy"].shape == (6, 57)
    assert env.height_scan_sensor.state_dict()["terrain_type"] == "wave"


def test_mac_anymal_c_rough_rollout_replay_is_deterministic():
    """Rollout and replay helpers should preserve the rough ANYmal-C trajectory for fixed actions."""

    cfg = MacAnymalCRoughEnvCfg(num_envs=4, seed=19, episode_length_s=0.5)
    actions = mx.zeros((cfg.num_envs, cfg.action_space), dtype=mx.float32)

    env_a = MacAnymalCRoughEnv(cfg)
    trace_a = rollout_env(env_a, actions, steps=6)

    env_b = MacAnymalCRoughEnv(cfg)
    trace_b = replay_actions(env_b, trace_a.actions)

    assert trace_a.summary() == trace_b.summary()
    assert mx.allclose(trace_a.observations[-1]["policy"], trace_b.observations[-1]["policy"]).item()
