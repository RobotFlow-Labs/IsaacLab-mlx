# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-native synthetic RGB/depth camera cartpole slices for the MLX/mac-sim path."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .cartpole import MacCartpoleEnv
from .env_cfgs import MacCartpoleDepthCameraEnvCfg, MacCartpoleRGBCameraEnvCfg


def _build_camera_grid(height: int, width: int) -> tuple[mx.array, mx.array]:
    x = mx.linspace(-1.0, 1.0, width, dtype=mx.float32).reshape((1, 1, width))
    y = mx.linspace(-1.0, 1.0, height, dtype=mx.float32).reshape((1, height, 1))
    return x, y


def _distance_to_segment(
    grid_x: mx.array,
    grid_y: mx.array,
    start_x: mx.array,
    start_y: mx.array,
    end_x: mx.array,
    end_y: mx.array,
) -> mx.array:
    start_x = start_x[:, None, None]
    start_y = start_y[:, None, None]
    end_x = end_x[:, None, None]
    end_y = end_y[:, None, None]
    seg_x = end_x - start_x
    seg_y = end_y - start_y
    proj = ((grid_x - start_x) * seg_x + (grid_y - start_y) * seg_y) / (mx.square(seg_x) + mx.square(seg_y) + 1e-6)
    proj = mx.clip(proj, 0.0, 1.0)
    closest_x = start_x + proj * seg_x
    closest_y = start_y + proj * seg_y
    return mx.sqrt(mx.square(grid_x - closest_x) + mx.square(grid_y - closest_y))


def _cartpole_pose(
    cart_pos: mx.array,
    pole_angle: mx.array,
    max_cart_pos: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    cart_center_x = mx.clip(cart_pos / max_cart_pos, -1.0, 1.0) * 0.72
    cart_center_y = mx.zeros_like(cart_center_x) + 0.46
    pole_tip_x = cart_center_x + 0.52 * mx.sin(pole_angle)
    pole_tip_y = cart_center_y - 0.52 * mx.cos(pole_angle)
    return cart_center_x, cart_center_y, pole_tip_x, pole_tip_y


def _render_cartpole_rgb_impl(
    grid_x: mx.array,
    grid_y: mx.array,
    cart_pos: mx.array,
    pole_angle: mx.array,
    max_cart_pos: float,
) -> mx.array:
    cart_center_x, cart_center_y, pole_tip_x, pole_tip_y = _cartpole_pose(cart_pos, pole_angle, max_cart_pos)

    rail_mask = mx.abs(grid_y - 0.56) <= 0.012
    cart_mask = (mx.abs(grid_x - cart_center_x[:, None, None]) <= 0.16) & (
        mx.abs(grid_y - cart_center_y[:, None, None]) <= 0.08
    )
    axle_mask = (
        mx.square(grid_x - cart_center_x[:, None, None]) + mx.square(grid_y - cart_center_y[:, None, None])
    ) <= 0.03**2
    pole_mask = _distance_to_segment(
        grid_x,
        grid_y,
        cart_center_x,
        cart_center_y,
        pole_tip_x,
        pole_tip_y,
    ) <= 0.028

    bg_y = (grid_y + 1.0) * 0.5
    red = mx.broadcast_to(0.14 + 0.10 * (1.0 - bg_y), cart_mask.shape)
    green = mx.broadcast_to(0.17 + 0.15 * (1.0 - bg_y), cart_mask.shape)
    blue = mx.broadcast_to(0.22 + 0.30 * (1.0 - bg_y), cart_mask.shape)

    red = mx.where(rail_mask, 0.62, red)
    green = mx.where(rail_mask, 0.56, green)
    blue = mx.where(rail_mask, 0.45, blue)

    red = mx.where(cart_mask, 0.88, red)
    green = mx.where(cart_mask, 0.46, green)
    blue = mx.where(cart_mask, 0.16, blue)

    red = mx.where(pole_mask, 0.94, red)
    green = mx.where(pole_mask, 0.92, green)
    blue = mx.where(pole_mask, 0.66, blue)

    red = mx.where(axle_mask, 0.18, red)
    green = mx.where(axle_mask, 0.22, green)
    blue = mx.where(axle_mask, 0.26, blue)

    image = mx.stack((red, green, blue), axis=-1).astype(mx.float32)
    return image - mx.mean(image, axis=(1, 2), keepdims=True)


def _render_cartpole_depth_impl(
    grid_x: mx.array,
    grid_y: mx.array,
    cart_pos: mx.array,
    pole_angle: mx.array,
    max_cart_pos: float,
    max_depth_m: float,
) -> mx.array:
    cart_center_x, cart_center_y, pole_tip_x, pole_tip_y = _cartpole_pose(cart_pos, pole_angle, max_cart_pos)

    rail_mask = mx.abs(grid_y - 0.56) <= 0.012
    cart_mask = (mx.abs(grid_x - cart_center_x[:, None, None]) <= 0.16) & (
        mx.abs(grid_y - cart_center_y[:, None, None]) <= 0.08
    )
    pole_mask = _distance_to_segment(
        grid_x,
        grid_y,
        cart_center_x,
        cart_center_y,
        pole_tip_x,
        pole_tip_y,
    ) <= 0.028

    depth = mx.zeros(cart_mask.shape, dtype=mx.float32)
    rail_depth = 3.5 + 0.2 * mx.abs(cart_center_x[:, None, None])
    cart_depth = 2.1 + 0.35 * mx.abs(cart_center_x[:, None, None])
    pole_depth = 1.8 + 0.25 * mx.abs(pole_tip_x[:, None, None] - cart_center_x[:, None, None])
    depth = mx.where(rail_mask, rail_depth, depth)
    depth = mx.where(cart_mask, cart_depth, depth)
    depth = mx.where(pole_mask, pole_depth, depth)
    return mx.clip(depth, 0.0, max_depth_m).astype(mx.float32)[..., None]


render_cartpole_rgb = mx.compile(_render_cartpole_rgb_impl)
render_cartpole_depth = mx.compile(_render_cartpole_depth_impl)


class MacCartpoleSyntheticCamera:
    """Deterministic analytic camera adapter for the cartpole sensor slices."""

    def __init__(self, *, mode: str, image_height: int, image_width: int, max_depth_m: float, max_cart_pos: float):
        self.mode = mode
        self.image_height = image_height
        self.image_width = image_width
        self.max_depth_m = max_depth_m
        self.max_cart_pos = max_cart_pos
        self.grid_x, self.grid_y = _build_camera_grid(image_height, image_width)

    def render(self, cart_pos: mx.array, pole_angle: mx.array) -> mx.array:
        if self.mode == "rgb":
            return render_cartpole_rgb(self.grid_x, self.grid_y, cart_pos, pole_angle, self.max_cart_pos)
        if self.mode == "depth":
            return render_cartpole_depth(
                self.grid_x,
                self.grid_y,
                cart_pos,
                pole_angle,
                self.max_cart_pos,
                self.max_depth_m,
            )
        raise ValueError(f"Unsupported camera mode: {self.mode}")

    def state_dict(self) -> dict[str, Any]:
        channels = 3 if self.mode == "rgb" else 1
        return {
            "backend": "mac-sensors",
            "implementation": "analytic-cartpole-camera",
            "camera_mode": self.mode,
            "image_shape": [self.image_height, self.image_width, channels],
            "max_depth_m": self.max_depth_m,
        }


class MacCartpoleCameraEnv(MacCartpoleEnv):
    """Cartpole environment with a deterministic synthetic RGB or depth camera observation."""

    cfg: MacCartpoleRGBCameraEnvCfg | MacCartpoleDepthCameraEnvCfg

    def __init__(self, cfg: MacCartpoleRGBCameraEnvCfg | MacCartpoleDepthCameraEnvCfg | None = None):
        self.cfg = cfg or MacCartpoleRGBCameraEnvCfg()
        self.camera_sensor = MacCartpoleSyntheticCamera(
            mode=self.cfg.camera_mode,
            image_height=self.cfg.image_height,
            image_width=self.cfg.image_width,
            max_depth_m=self.cfg.camera_max_depth_m,
            max_cart_pos=self.cfg.max_cart_pos,
        )
        super().__init__(self.cfg)
        self.single_observation_space = {"policy": self.cfg.observation_space}

    @property
    def camera_mode(self) -> str:
        return self.cfg.camera_mode

    def _get_observations(self) -> dict[str, mx.array]:
        joint_pos, _ = self._joint_state()
        cart_pos = joint_pos[:, 0]
        pole_angle = joint_pos[:, 1]
        image = self.camera_sensor.render(cart_pos, pole_angle)
        return {"policy": image}
