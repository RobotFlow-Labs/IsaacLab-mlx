# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""macOS external camera discovery and raw capture helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
import time
from typing import Any


DEVICE_RE = re.compile(r"\[AVFoundation indev @ .+?\] \[(?P<index>\d+)\] (?P<name>.+)")
MODE_RE = re.compile(
    r"\[avfoundation @ .+?\]\s+(?P<width>\d+)x(?P<height>\d+)@\[(?P<min_fps>[0-9.]+)\s+(?P<max_fps>[0-9.]+)\]fps"
)


class MacCameraCaptureError(RuntimeError):
    """Raised when the macOS external camera path fails explicitly."""


@dataclass(frozen=True)
class MacCameraMode:
    """A supported AVFoundation video mode."""

    width: int
    height: int
    min_fps: float
    max_fps: float


@dataclass(frozen=True)
class MacCameraDevice:
    """A discoverable macOS camera device."""

    index: int
    name: str
    unique_id: str | None = None
    modes: tuple[MacCameraMode, ...] = ()

    @property
    def is_stereolabs(self) -> bool:
        return "zed" in self.name.lower() or "stereolab" in self.name.lower()

    def state_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "unique_id": self.unique_id,
            "is_stereolabs": self.is_stereolabs,
            "modes": [asdict(mode) for mode in self.modes],
        }


def _run_command(args: list[str], *, timeout_s: float = 15.0) -> str:
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def parse_avfoundation_device_list(output: str) -> tuple[MacCameraDevice, ...]:
    """Parse the AVFoundation device list emitted by ffmpeg."""

    devices: list[MacCameraDevice] = []
    for line in output.splitlines():
        match = DEVICE_RE.search(line)
        if not match:
            continue
        name = match.group("name").strip()
        if name.startswith("Capture screen"):
            continue
        devices.append(MacCameraDevice(index=int(match.group("index")), name=name))
    return tuple(devices)


def parse_avfoundation_supported_modes(output: str) -> tuple[MacCameraMode, ...]:
    """Parse supported AVFoundation modes from an ffmpeg probe error."""

    modes: list[MacCameraMode] = []
    seen: set[tuple[int, int, float, float]] = set()
    for line in output.splitlines():
        match = MODE_RE.search(line)
        if not match:
            continue
        mode = MacCameraMode(
            width=int(match.group("width")),
            height=int(match.group("height")),
            min_fps=float(match.group("min_fps")),
            max_fps=float(match.group("max_fps")),
        )
        key = (mode.width, mode.height, mode.min_fps, mode.max_fps)
        if key in seen:
            continue
        seen.add(key)
        modes.append(mode)
    return tuple(modes)


def list_avfoundation_cameras(*, ffmpeg_bin: str = "ffmpeg", timeout_s: float = 10.0) -> tuple[MacCameraDevice, ...]:
    """List visible AVFoundation cameras through ffmpeg."""

    output = _run_command(
        [ffmpeg_bin, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        timeout_s=timeout_s,
    )
    return parse_avfoundation_device_list(output)


def probe_avfoundation_camera_modes(
    device_index: int,
    *,
    ffmpeg_bin: str = "ffmpeg",
    timeout_s: float = 10.0,
) -> tuple[MacCameraMode, ...]:
    """Probe supported modes for a single AVFoundation device.

    We intentionally request an invalid video size so ffmpeg prints the supported modes
    without trying to capture a real frame.
    """

    output = _run_command(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-video_size",
            "1x1",
            "-i",
            f"{device_index}:none",
        ],
        timeout_s=timeout_s,
    )
    return parse_avfoundation_supported_modes(output)


def discover_external_cameras(
    *,
    ffmpeg_bin: str = "ffmpeg",
    include_modes: bool = True,
    timeout_s: float = 10.0,
) -> tuple[MacCameraDevice, ...]:
    """Discover macOS external cameras, optionally enriching them with supported modes."""

    devices = list_avfoundation_cameras(ffmpeg_bin=ffmpeg_bin, timeout_s=timeout_s)
    if not include_modes:
        return devices
    enriched: list[MacCameraDevice] = []
    for device in devices:
        modes = probe_avfoundation_camera_modes(device.index, ffmpeg_bin=ffmpeg_bin, timeout_s=timeout_s)
        enriched.append(MacCameraDevice(index=device.index, name=device.name, unique_id=device.unique_id, modes=modes))
    return tuple(enriched)


def capture_avfoundation_raw_frame(
    *,
    device_index: int,
    output_path: str | Path,
    width: int,
    height: int,
    framerate: int | float = 30,
    ffmpeg_bin: str = "ffmpeg",
    timeout_s: float = 20.0,
    pixel_format: str = "yuyv422",
) -> dict[str, Any]:
    """Capture a single raw frame from an AVFoundation camera into a side-by-side dump."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    completed = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-y",
            "-f",
            "avfoundation",
            "-framerate",
            str(framerate),
            "-video_size",
            f"{width}x{height}",
            "-i",
            f"{device_index}:none",
            "-frames:v",
            "1",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            str(output_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    if completed.returncode != 0:
        if "Could not lock device for configuration" in stderr:
            raise MacCameraCaptureError(
                "AVFoundation could not lock the camera device. Another app or agent is likely holding the camera."
            )
        raise MacCameraCaptureError(stderr or stdout or f"ffmpeg exited with code {completed.returncode}")

    metadata = {
        "device_index": device_index,
        "width": width,
        "height": height,
        "channels": 2,
        "pixel_format": pixel_format,
        "timestamp": started_ns,
    }
    sidecar_path = Path(f"{output_file}.txt")
    sidecar_path.write_text(
        "\n".join(f"{key}={value}" for key, value in metadata.items()) + "\n",
        encoding="utf-8",
    )
    return {
        "output_path": str(output_file),
        "metadata_path": str(sidecar_path),
        "device_index": device_index,
        "width": width,
        "height": height,
        "pixel_format": pixel_format,
        "timestamp": started_ns,
    }
