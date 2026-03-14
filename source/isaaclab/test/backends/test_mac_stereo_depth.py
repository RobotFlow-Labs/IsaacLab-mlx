# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np

from isaaclab.backends.test_utils import require_mlx_runtime

require_mlx_runtime()

from isaaclab.backends.mac_sim import (  # noqa: E402
    compute_disparity_absdiff,
    disparity_to_depth_mm,
    load_raw_stereo_frame,
    normalize_depth_for_preview,
    normalize_disparity_for_preview,
    stereo_luma_from_yuyv,
    stereo_rgb_from_yuyv,
    yuyv_to_rgb,
)


def test_load_raw_stereo_frame_reads_sidecar_metadata(tmp_path: Path):
    frame_path = tmp_path / "frame.raw"
    metadata_path = Path(f"{frame_path}.txt")
    payload = bytes(range(16))
    frame_path.write_bytes(payload)
    metadata_path.write_text("width=4\nheight=2\nchannels=2\npixel_format=yuyv422\ntimestamp=123\n", encoding="utf-8")

    frame = load_raw_stereo_frame(frame_path)

    assert frame.width == 4
    assert frame.height == 2
    assert frame.pixel_format == "yuyv422"
    assert frame.timestamp == 123
    assert frame.yuyv.shape == (2, 4, 2)


def test_split_stereo_luma_uses_y_channel():
    frame = np.array(
        [
            [[10, 101], [20, 102], [30, 103], [40, 104]],
            [[50, 105], [60, 106], [70, 107], [80, 108]],
        ],
        dtype=np.uint8,
    )

    left_luma, right_luma = stereo_luma_from_yuyv(frame)

    np.testing.assert_array_equal(left_luma, np.array([[10, 20], [50, 60]], dtype=np.uint8))
    np.testing.assert_array_equal(right_luma, np.array([[30, 40], [70, 80]], dtype=np.uint8))


def test_yuyv_to_rgb_preserves_shape():
    frame = np.array(
        [
            [[100, 128], [110, 128], [120, 128], [130, 128]],
            [[140, 128], [150, 128], [160, 128], [170, 128]],
        ],
        dtype=np.uint8,
    )

    rgb = yuyv_to_rgb(frame)
    left_rgb, right_rgb = stereo_rgb_from_yuyv(frame)

    assert rgb.shape == (2, 4, 3)
    assert left_rgb.shape == (2, 2, 3)
    assert right_rgb.shape == (2, 2, 3)


def test_absdiff_disparity_recovers_constant_horizontal_shift():
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, size=(24, 48), dtype=np.uint8)
    disparity = 4
    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]

    result = compute_disparity_absdiff(left, right, max_disparity=8, window_size=1)
    stable_region = result[:, disparity + 2 : -2]

    assert stable_region.size > 0
    assert np.median(stable_region) == disparity


def test_disparity_and_depth_preview_helpers():
    disparity = np.array([[2.0, 4.0], [8.0, 0.25]], dtype=np.float32)
    depth = disparity_to_depth_mm(disparity, focal_length_px=700.0, baseline_mm=120.0)
    disparity_vis = normalize_disparity_for_preview(disparity, max_disparity=8)
    depth_vis = normalize_depth_for_preview(depth, max_depth_mm=42000.0)

    np.testing.assert_allclose(depth[0, 0], 42000.0)
    np.testing.assert_allclose(depth[0, 1], 21000.0)
    np.testing.assert_allclose(depth[1, 0], 10500.0)
    assert np.isnan(depth[1, 1])
    assert disparity_vis.dtype == np.uint8
    assert depth_vis.dtype == np.uint8
