# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the mac-native cartpole showcase spaces."""

from __future__ import annotations

import pytest

from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()
gym = pytest.importorskip("gymnasium")

from isaaclab.backends.mac_sim import SHOWCASE_CFGS, MacCartpoleShowcaseEnv  # noqa: E402


def _batched_action(space: gym.Space, num_envs: int):
    if isinstance(space, gym.spaces.Box):
        return [space.sample().tolist() for _ in range(num_envs)]
    if isinstance(space, gym.spaces.Discrete):
        return [space.sample() for _ in range(num_envs)]
    if isinstance(space, gym.spaces.MultiDiscrete):
        return [space.sample().tolist() for _ in range(num_envs)]
    raise TypeError(f"Unsupported action space for test generation: {type(space)}")


def _assert_policy_obs_matches_space(obs, space: gym.Space, num_envs: int):
    if isinstance(space, gym.spaces.Box):
        assert obs.shape == (num_envs, *space.shape)
        return
    if isinstance(space, gym.spaces.Discrete):
        assert obs.shape == (num_envs,)
        return
    if isinstance(space, gym.spaces.MultiDiscrete):
        assert obs.shape == (num_envs, len(space.nvec))
        return
    if isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs, tuple)
        assert len(obs) == len(space.spaces)
        for item, item_space in zip(obs, space.spaces, strict=True):
            assert item.shape == (num_envs, *item_space.shape)
        return
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(obs, dict)
        assert set(obs.keys()) == set(space.spaces.keys())
        for key, item_space in space.spaces.items():
            assert obs[key].shape == (num_envs, *item_space.shape)
        return
    raise TypeError(f"Unsupported observation space for assertions: {type(space)}")


@pytest.mark.parametrize("cfg_type", SHOWCASE_CFGS, ids=lambda cfg_type: cfg_type.__name__)
def test_mac_cartpole_showcase_step_shapes(cfg_type):
    """Every showcase config should reset and step through the mac-native cartpole backend."""
    cfg = cfg_type(num_envs=8, seed=17, episode_length_s=0.3)
    env = MacCartpoleShowcaseEnv(cfg)

    obs, extras = env.reset()

    assert extras == {}
    _assert_policy_obs_matches_space(obs["policy"], cfg.observation_space, cfg.num_envs)

    next_obs, reward, terminated, truncated, step_extras = env.step(_batched_action(cfg.action_space, cfg.num_envs))

    _assert_policy_obs_matches_space(next_obs["policy"], cfg.observation_space, cfg.num_envs)
    assert reward.shape == (cfg.num_envs,)
    assert terminated.shape == (cfg.num_envs,)
    assert truncated.shape == (cfg.num_envs,)
    assert isinstance(step_extras, dict)
