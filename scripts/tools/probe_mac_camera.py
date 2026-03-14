# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe macOS external cameras and optionally capture one raw stereo frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.backends.mac_sim import CAPTURE_BACKENDS, capture_mac_camera_raw_frame, discover_external_cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe macOS AVFoundation cameras for backend-local MLX work.")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-modes", action="store_true")
    parser.add_argument("--capture-device-index", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=None)
    parser.add_argument("--capture-height", type=int, default=None)
    parser.add_argument("--capture-framerate", type=float, default=30.0)
    parser.add_argument("--capture-output", type=Path, default=None)
    parser.add_argument("--capture-timeout-s", type=float, default=60.0)
    parser.add_argument("--capture-backend", choices=CAPTURE_BACKENDS, default="auto")
    parser.add_argument("--zed-sdk-mlx-repo", type=Path, default=None)
    args = parser.parse_args()

    devices = discover_external_cameras(
        ffmpeg_bin=args.ffmpeg_bin,
        include_modes=not args.skip_modes,
    )
    payload: dict[str, object] = {
        "device_count": len(devices),
        "devices": [device.state_dict() for device in devices],
    }

    if args.capture_device_index is not None:
        if args.capture_width is None or args.capture_height is None or args.capture_output is None:
            raise SystemExit("--capture-width, --capture-height, and --capture-output are required with --capture-device-index")
        selected_device = next((device for device in devices if device.index == args.capture_device_index), None)
        payload["capture"] = capture_mac_camera_raw_frame(
            device_index=args.capture_device_index,
            output_path=args.capture_output,
            width=args.capture_width,
            height=args.capture_height,
            framerate=args.capture_framerate,
            ffmpeg_bin=args.ffmpeg_bin,
            timeout_s=args.capture_timeout_s,
            capture_backend=args.capture_backend,
            zed_sdk_mlx_repo=args.zed_sdk_mlx_repo,
            device=selected_device,
        )

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
