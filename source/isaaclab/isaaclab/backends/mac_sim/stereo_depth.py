# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal MLX stereo/depth helpers for backend-local macOS camera work."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RawStereoFrame:
    """A side-by-side YUYV stereo frame dump plus its metadata."""

    path: Path
    width: int
    height: int
    channels: int
    pixel_format: str
    frame_id: int
    timestamp: int
    serial_number: int | None
    yuyv: np.ndarray


def _metadata_path(input_path: Path) -> Path:
    return Path(f"{input_path}.txt")


def load_frame_metadata(input_path: Path) -> dict[str, str]:
    """Load the key=value sidecar metadata written next to a raw frame dump."""

    values: dict[str, str] = {}
    metadata_path = _metadata_path(input_path)
    if not metadata_path.exists():
        return values
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        values[key] = raw_value
    return values


def load_raw_stereo_frame(input_path: str | Path, *, width: int | None = None, height: int | None = None) -> RawStereoFrame:
    """Load a raw YUYV stereo dump captured from the backend-local camera path."""

    frame_path = Path(input_path)
    metadata = load_frame_metadata(frame_path)
    resolved_width = width or (int(metadata["width"]) if "width" in metadata else None)
    resolved_height = height or (int(metadata["height"]) if "height" in metadata else None)
    if resolved_width is None or resolved_height is None:
        raise ValueError("width and height are required when no metadata sidecar is present")

    payload = frame_path.read_bytes()
    expected_bytes = int(resolved_width) * int(resolved_height) * 2
    if len(payload) != expected_bytes:
        raise ValueError(f"unexpected frame size: got {len(payload)} bytes, expected {expected_bytes} for YUYV422")

    yuyv = np.frombuffer(payload, dtype=np.uint8).reshape((int(resolved_height), int(resolved_width), 2))
    return RawStereoFrame(
        path=frame_path,
        width=int(resolved_width),
        height=int(resolved_height),
        channels=int(metadata.get("channels", "2")),
        pixel_format=str(metadata.get("pixel_format", "yuyv422")),
        frame_id=int(metadata.get("frame_id", "0")),
        timestamp=int(metadata.get("timestamp", "0")),
        serial_number=int(metadata["serial_number"]) if "serial_number" in metadata else None,
        yuyv=yuyv,
    )


def split_stereo_yuyv(frame_yuyv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side YUYV frame into left/right YUYV images."""

    if frame_yuyv.ndim != 3 or frame_yuyv.shape[2] != 2:
        raise ValueError("expected a HxWx2 YUYV frame")
    width = int(frame_yuyv.shape[1])
    if width % 2 != 0:
        raise ValueError("stereo YUYV frame width must be even")
    mid = width // 2
    return frame_yuyv[:, :mid].copy(), frame_yuyv[:, mid:].copy()


def stereo_luma_from_yuyv(frame_yuyv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract left/right luma images from a side-by-side YUYV frame."""

    left_yuyv, right_yuyv = split_stereo_yuyv(frame_yuyv)
    return left_yuyv[:, :, 0].copy(), right_yuyv[:, :, 0].copy()


def _yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    y_f = y.astype(np.float32)
    u_f = u.astype(np.float32) - 128.0
    v_f = v.astype(np.float32) - 128.0
    r = np.clip(y_f + 1.402 * v_f, 0.0, 255.0)
    g = np.clip(y_f - 0.344136 * u_f - 0.714136 * v_f, 0.0, 255.0)
    b = np.clip(y_f + 1.772 * u_f, 0.0, 255.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def yuyv_to_rgb(frame_yuyv: np.ndarray) -> np.ndarray:
    """Convert a YUYV image to RGB."""

    if frame_yuyv.ndim != 3 or frame_yuyv.shape[2] != 2:
        raise ValueError("expected a HxWx2 YUYV frame")
    raw = frame_yuyv.reshape((frame_yuyv.shape[0], frame_yuyv.shape[1] // 2, 4))
    y0 = raw[:, :, 0]
    u = raw[:, :, 1]
    y1 = raw[:, :, 2]
    v = raw[:, :, 3]
    rgb0 = _yuv_to_rgb(y0, u, v)
    rgb1 = _yuv_to_rgb(y1, u, v)
    output = np.empty((frame_yuyv.shape[0], frame_yuyv.shape[1], 3), dtype=np.uint8)
    output[:, 0::2] = rgb0
    output[:, 1::2] = rgb1
    return output


def stereo_rgb_from_yuyv(frame_yuyv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a side-by-side YUYV stereo frame to left/right RGB images."""

    rgb = yuyv_to_rgb(frame_yuyv)
    mid = rgb.shape[1] // 2
    return rgb[:, :mid].copy(), rgb[:, mid:].copy()


def _to_unit_float(image: np.ndarray) -> mx.array:
    if image.ndim != 2:
        raise ValueError("expected a single-channel HxW image")
    return mx.array(image, dtype=mx.float32) / 255.0


@lru_cache(maxsize=16)
def _avg_pool(window_size: int) -> nn.AvgPool2d:
    return nn.AvgPool2d(kernel_size=window_size, stride=1, padding=window_size // 2)


def _box_filter_2d(cost: mx.array, window_size: int) -> mx.array:
    if window_size <= 1:
        return cost
    pool = _avg_pool(window_size)
    if cost.ndim == 2:
        return pool(cost[None, :, :, None])[0, :, :, 0]
    if cost.ndim == 3:
        return pool(cost[None, :, :, :])[0]
    raise ValueError(f"expected 2D or 3D cost tensor, got {cost.ndim}D")


def _shift_volume(image: mx.array, *, max_disparity: int) -> tuple[mx.array, mx.array]:
    width = int(image.shape[1])
    max_supported_disparity = min(max_disparity, width)
    disparities = mx.arange(max_supported_disparity, dtype=mx.int32)[None, :]
    x_coords = mx.arange(width, dtype=mx.int32)[:, None]
    source_x = x_coords - disparities
    valid = ((source_x >= 0) & (source_x < width)).astype(mx.float32)
    indices = mx.clip(source_x, 0, width - 1)
    shifted = mx.take(image, indices, axis=1)
    valid = mx.broadcast_to(valid[None, :, :], shifted.shape)
    return shifted, valid


def _compute_disparity_absdiff_mlx(left: mx.array, right: mx.array, *, max_disparity: int, window_size: int) -> mx.array:
    shifted, valid = _shift_volume(right, max_disparity=max_disparity)
    cost = mx.abs(left[:, :, None] - shifted)
    cost = _box_filter_2d(cost, window_size)
    invalid_cost = 2.0
    volume = cost + (1.0 - valid) * invalid_cost
    return mx.argmin(volume, axis=-1).astype(mx.float32)


@lru_cache(maxsize=16)
def _compiled_absdiff_kernel(max_disparity: int, window_size: int):
    return mx.compile(
        lambda left, right: _compute_disparity_absdiff_mlx(
            left,
            right,
            max_disparity=max_disparity,
            window_size=window_size,
        )
    )


def compute_disparity_absdiff(
    left_luma: np.ndarray,
    right_luma: np.ndarray,
    *,
    max_disparity: int = 64,
    window_size: int = 5,
) -> np.ndarray:
    """Compute a minimal MLX disparity map from rectified or near-rectified luma pairs."""

    if left_luma.shape != right_luma.shape:
        raise ValueError("left/right image shapes must match")
    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    left = _to_unit_float(left_luma)
    right = _to_unit_float(right_luma)
    disparity = _compiled_absdiff_kernel(max_disparity, window_size)(left, right)
    mx.eval(disparity)
    return np.array(disparity, dtype=np.float32)


def disparity_to_depth_mm(disparity: np.ndarray, *, focal_length_px: float, baseline_mm: float) -> np.ndarray:
    """Convert disparity pixels to depth in millimeters."""

    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0.5
    depth[valid] = (float(focal_length_px) * float(baseline_mm)) / disparity[valid]
    return depth


def normalize_disparity_for_preview(disparity: np.ndarray, max_disparity: int) -> np.ndarray:
    """Normalize disparity to an 8-bit preview image."""

    scale = 255.0 / max(float(max_disparity), 1.0)
    return np.clip(disparity * scale, 0.0, 255.0).astype(np.uint8)


def normalize_depth_for_preview(depth_mm: np.ndarray, *, max_depth_mm: float) -> np.ndarray:
    """Normalize depth to an 8-bit preview image, with nearer pixels brighter."""

    clipped = np.nan_to_num(depth_mm, nan=max_depth_mm, posinf=max_depth_mm)
    normalized = 1.0 - np.clip(clipped / max(float(max_depth_mm), 1.0), 0.0, 1.0)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def save_preview_png(image: np.ndarray, path: str | Path) -> Path:
    """Save a grayscale or RGB preview PNG."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return output_path
