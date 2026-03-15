# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the backend-local MLX stereo/depth smoke path on a raw stereo dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaaclab.backends.mac_sim import (
    compute_disparity_absdiff,
    disparity_to_depth_mm,
    load_raw_stereo_frame,
    normalize_depth_for_preview,
    normalize_disparity_for_preview,
    save_preview_png,
    stereo_luma_from_yuyv,
    stereo_rgb_from_yuyv,
    validate_raw_capture_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MLX stereo/depth smoke path on a raw backend-local frame dump.")
    parser.add_argument("input_path", type=Path, help="Raw YUYV dump path from probe_mac_camera.py")
    parser.add_argument("output_dir", type=Path, help="Directory for preview PNGs")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--max-disparity", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--fx", type=float, help="Optional focal length in pixels for depth conversion")
    parser.add_argument("--baseline-mm", type=float, help="Optional stereo baseline in millimeters")
    parser.add_argument("--max-depth-mm", type=float, default=5000.0)
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    metadata_path = Path(f"{args.input_path}.txt")
    capture_metadata = None
    if metadata_path.exists():
        capture_metadata = validate_raw_capture_artifacts(args.input_path, metadata_path)

    frame = load_raw_stereo_frame(args.input_path, width=args.width, height=args.height)
    left_luma, right_luma = stereo_luma_from_yuyv(frame.yuyv)
    left_rgb, right_rgb = stereo_rgb_from_yuyv(frame.yuyv)
    disparity = compute_disparity_absdiff(
        left_luma,
        right_luma,
        max_disparity=args.max_disparity,
        window_size=args.window_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_rgb_path = save_preview_png(left_rgb, args.output_dir / "left_rgb.png")
    right_rgb_path = save_preview_png(right_rgb, args.output_dir / "right_rgb.png")
    disparity_path = save_preview_png(
        normalize_disparity_for_preview(disparity, args.max_disparity),
        args.output_dir / "disparity.png",
    )
    summary_out = args.summary_out or (args.output_dir / "summary.json")
    summary = {
        "input_path": str(args.input_path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "validated_capture": capture_metadata is not None,
        "frame": {
            "width": frame.width,
            "height": frame.height,
            "channels": frame.channels,
            "pixel_format": frame.pixel_format,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "serial_number": frame.serial_number,
        },
        "capture_metadata": capture_metadata,
        "left_rgb_path": str(left_rgb_path),
        "right_rgb_path": str(right_rgb_path),
        "disparity_path": str(disparity_path),
        "disparity_shape": list(disparity.shape),
        "disparity_nonzero_ratio": float(np.mean(disparity > 0.0)),
    }

    print(f"left_rgb={left_rgb_path}")
    print(f"right_rgb={right_rgb_path}")
    print(f"disparity_png={disparity_path}")
    print(f"disparity_shape={disparity.shape}")

    if args.fx is not None or args.baseline_mm is not None:
        if args.fx is None or args.baseline_mm is None:
            raise SystemExit("--fx and --baseline-mm must be provided together")
        depth = disparity_to_depth_mm(disparity, focal_length_px=args.fx, baseline_mm=args.baseline_mm)
        depth_path = save_preview_png(
            normalize_depth_for_preview(depth, max_depth_mm=args.max_depth_mm),
            args.output_dir / "depth.png",
        )
        summary["depth_path"] = str(depth_path)
        summary["depth_shape"] = list(depth.shape)
        summary["depth_finite_ratio"] = float(np.mean(np.isfinite(depth)))
        finite_depth = depth[np.isfinite(depth)]
        summary["depth_positive_ratio"] = float(np.mean((depth > 0.0) & np.isfinite(depth)))
        summary["depth_mm_min"] = float(np.min(finite_depth)) if finite_depth.size else None
        summary["depth_mm_max"] = float(np.max(finite_depth)) if finite_depth.size else None
        summary["depth_mm_mean"] = float(np.mean(finite_depth)) if finite_depth.size else None
        print(f"depth_png={depth_path}")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary_json={summary_out}")


if __name__ == "__main__":
    main()
