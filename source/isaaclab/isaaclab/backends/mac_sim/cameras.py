# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""macOS external camera discovery and raw capture helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any


DEVICE_RE = re.compile(r"\[AVFoundation indev @ .+?\] \[(?P<index>\d+)\] (?P<name>.+)")
VIDEO_SECTION_RE = re.compile(r"\[AVFoundation indev @ .+?\] AVFoundation video devices:")
AUDIO_SECTION_RE = re.compile(r"\[AVFoundation indev @ .+?\] AVFoundation audio devices:")
MODE_RE = re.compile(
    r"\[avfoundation @ .+?\]\s+(?P<width>\d+)x(?P<height>\d+)@\[(?P<min_fps>[0-9.]+)\s+(?P<max_fps>[0-9.]+)\]fps"
)
ZED_SDK_MLX_ENV = "ZED_SDK_MLX_REPO"
CAPTURE_BACKEND_AUTO = "auto"
CAPTURE_BACKEND_AVFOUNDATION = "avfoundation"
CAPTURE_BACKEND_ZED_DIRECT = "zed-sdk-mlx-direct"
CAPTURE_BACKEND_ZED_TERMINAL = "zed-sdk-mlx-terminal"
CAPTURE_BACKENDS = (
    CAPTURE_BACKEND_AUTO,
    CAPTURE_BACKEND_AVFOUNDATION,
    CAPTURE_BACKEND_ZED_DIRECT,
    CAPTURE_BACKEND_ZED_TERMINAL,
)
ZED_SDK_MLX_CAPTURE_WIDTH = 2560
ZED_SDK_MLX_CAPTURE_HEIGHT = 720
ZED_SDK_MLX_CAPTURE_FRAMERATE = 30.0
_ZED_CAPTURE_APP_RELATIVE_BIN = Path("build/zed-capture-diagnostics/ZEDCaptureDiagnostics.app/Contents/MacOS/ZEDCaptureDiagnostics")
_INT_CAPTURE_METADATA_FIELDS = {"width", "height", "channels", "frame_id", "timestamp", "serial_number", "device_index"}
_FLOAT_CAPTURE_METADATA_FIELDS = {"framerate"}


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


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _zed_capture_app_bin(repo_path: Path) -> Path:
    return repo_path / _ZED_CAPTURE_APP_RELATIVE_BIN


def _normalize_key_value_metadata(value: str, *, key: str) -> int | float | str:
    if key in _INT_CAPTURE_METADATA_FIELDS:
        return int(value)
    if key in _FLOAT_CAPTURE_METADATA_FIELDS:
        return float(value)
    return value


def parse_capture_metadata_text(text: str) -> dict[str, Any]:
    """Parse key=value metadata emitted alongside raw frame dumps."""

    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        metadata[key] = _normalize_key_value_metadata(raw_value, key=key)
    return metadata


def read_capture_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Read and parse a raw-frame metadata sidecar."""

    path = Path(metadata_path)
    return parse_capture_metadata_text(path.read_text(encoding="utf-8"))


def validate_raw_capture_artifacts(
    output_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    """Validate that a raw capture and its sidecar agree on frame shape."""

    output_file = Path(output_path)
    sidecar_path = Path(metadata_path)
    if not output_file.exists():
        raise MacCameraCaptureError(f"raw capture output is missing: {output_file}")
    if not sidecar_path.exists():
        raise MacCameraCaptureError(f"raw capture metadata sidecar is missing: {sidecar_path}")

    metadata = read_capture_metadata(sidecar_path)
    missing_fields = sorted(
        field for field in ("width", "height", "channels", "timestamp") if field not in metadata
    )
    if missing_fields:
        raise MacCameraCaptureError(
            f"raw capture metadata is missing required fields: {', '.join(missing_fields)}"
        )

    expected_bytes = int(metadata["width"]) * int(metadata["height"]) * int(metadata["channels"])
    actual_bytes = output_file.stat().st_size
    if expected_bytes != actual_bytes:
        raise MacCameraCaptureError(
            "raw capture size does not match metadata: "
            f"expected {expected_bytes} bytes but found {actual_bytes} bytes"
        )
    return metadata


def resolve_zed_sdk_mlx_repo(repo_path: str | Path | None = None) -> Path | None:
    """Resolve the local zed-sdk-mlx checkout used for Terminal-hosted ZED capture."""

    candidate = repo_path or os.environ.get(ZED_SDK_MLX_ENV)
    if candidate is None:
        return None

    resolved = Path(candidate).expanduser().resolve()
    if not resolved.exists():
        raise MacCameraCaptureError(f"zed-sdk-mlx repo does not exist: {resolved}")
    if not (resolved / "Makefile").is_file():
        raise MacCameraCaptureError(f"zed-sdk-mlx repo is missing Makefile: {resolved}")
    if not (resolved / "tools" / "zed_capture_app.mm").is_file():
        raise MacCameraCaptureError(f"zed-sdk-mlx repo is missing tools/zed_capture_app.mm: {resolved}")
    return resolved


def _ensure_zed_capture_constraints(*, width: int, height: int, framerate: int | float) -> None:
    if width != ZED_SDK_MLX_CAPTURE_WIDTH or height != ZED_SDK_MLX_CAPTURE_HEIGHT:
        raise MacCameraCaptureError(
            "zed-sdk-mlx capture currently supports only "
            f"{ZED_SDK_MLX_CAPTURE_WIDTH}x{ZED_SDK_MLX_CAPTURE_HEIGHT} raw dumps"
        )
    if float(framerate) != ZED_SDK_MLX_CAPTURE_FRAMERATE:
        raise MacCameraCaptureError(
            "zed-sdk-mlx capture currently supports only "
            f"{int(ZED_SDK_MLX_CAPTURE_FRAMERATE)} FPS raw dumps"
        )


def _format_zed_capture_failure(detail: str, *, source: str) -> str:
    if "camera_permission_missing" in detail:
        return (
            "zed-sdk-mlx capture failed because camera access is not attributed to the current host. "
            "Use the Terminal-hosted backend or grant Camera permission to the host app. "
            f"Details from {source}: {detail.strip()}"
        )
    if "Could not lock device for configuration" in detail:
        return "The ZED camera is busy. Another app or agent is likely holding the device."
    if "no_frame_received" in detail:
        return f"zed-sdk-mlx did not receive a frame before timing out. Details from {source}: {detail.strip()}"
    if "initializeVideo failed" in detail:
        return f"zed-sdk-mlx could not initialize the requested ZED device. Details from {source}: {detail.strip()}"
    return detail.strip() or f"zed-sdk-mlx capture failed without a detailed {source} message"


def _run_completed_command(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    timeout_s: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _build_zed_capture_app(repo_path: Path, *, timeout_s: float = 180.0) -> Path:
    app_bin = _zed_capture_app_bin(repo_path)
    if app_bin.exists():
        return app_bin
    completed = _run_completed_command(["make", "build-capture-app"], cwd=repo_path, timeout_s=timeout_s)
    if completed.returncode != 0:
        detail = ((completed.stdout or "") + (completed.stderr or "")).strip()
        raise MacCameraCaptureError(f"failed to build zed-sdk-mlx capture app: {detail}")
    if not app_bin.exists():
        raise MacCameraCaptureError(f"zed-sdk-mlx capture app build completed but binary is missing: {app_bin}")
    return app_bin


def _write_terminal_capture_script(
    *,
    repo_path: Path,
    app_bin: Path,
    output_path: Path,
    log_path: Path,
    status_path: Path,
    device_index: int,
) -> Path:
    script_path = status_path.with_suffix(".command")
    script_text = f"""#!/bin/bash
set -euo pipefail

REPO_DIR={shlex.quote(str(repo_path))}
APP_BIN={shlex.quote(str(app_bin))}
OUTPUT_PATH={shlex.quote(str(output_path))}
LOG_PATH={shlex.quote(str(log_path))}
STATUS_PATH={shlex.quote(str(status_path))}

cd "$REPO_DIR"
rm -f "$STATUS_PATH" "$OUTPUT_PATH" "${{OUTPUT_PATH}}.txt"

set +e
{{
  echo "repo_dir=$REPO_DIR"
  date
  "$APP_BIN" --device {device_index} "$OUTPUT_PATH"
}} 2>&1 | tee "$LOG_PATH"
status=${{PIPESTATUS[0]}}
set -e

printf '%s\\n' "$status" > "$STATUS_PATH"
echo "log_path=$LOG_PATH"
echo "status_path=$STATUS_PATH"
echo "exit_code=$status"
exit "$status"
"""
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _open_terminal_script(script_path: Path, *, timeout_s: float = 10.0) -> None:
    completed = _run_completed_command(["open", "-a", "Terminal", str(script_path)], timeout_s=timeout_s)
    if completed.returncode != 0:
        detail = ((completed.stdout or "") + (completed.stderr or "")).strip()
        raise MacCameraCaptureError(f"failed to launch Terminal-hosted ZED capture script: {detail}")


def _wait_for_terminal_capture(
    *,
    output_path: Path,
    status_path: Path,
    log_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if status_path.exists():
            raw_status = status_path.read_text(encoding="utf-8").strip() or "1"
            try:
                exit_code = int(raw_status)
            except ValueError as exc:
                raise MacCameraCaptureError(f"invalid zed-sdk-mlx terminal status file contents: {raw_status!r}") from exc
            if exit_code != 0:
                raise MacCameraCaptureError(_format_zed_capture_failure(_tail_text(log_path), source=str(log_path)))
            if not output_path.exists():
                raise MacCameraCaptureError("zed-sdk-mlx terminal capture exited successfully but did not write an output frame")
            return
        time.sleep(0.25)
    raise MacCameraCaptureError(
        "timed out waiting for Terminal-hosted zed-sdk-mlx capture to finish. "
        + _format_zed_capture_failure(_tail_text(log_path), source=str(log_path))
    )


def parse_avfoundation_device_list(output: str) -> tuple[MacCameraDevice, ...]:
    """Parse the AVFoundation device list emitted by ffmpeg."""

    devices: list[MacCameraDevice] = []
    in_video_section = False
    for line in output.splitlines():
        if VIDEO_SECTION_RE.search(line):
            in_video_section = True
            continue
        if AUDIO_SECTION_RE.search(line):
            in_video_section = False
            continue
        if not in_video_section:
            continue
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
    validate_raw_capture_artifacts(output_file, sidecar_path)
    return {
        "output_path": str(output_file),
        "metadata_path": str(sidecar_path),
        "device_index": device_index,
        "width": width,
        "height": height,
        "pixel_format": pixel_format,
        "timestamp": started_ns,
    }


def capture_zed_sdk_mlx_raw_frame(
    *,
    device_index: int,
    output_path: str | Path,
    width: int,
    height: int,
    framerate: int | float = ZED_SDK_MLX_CAPTURE_FRAMERATE,
    repo_path: str | Path | None = None,
    launch_mode: str = "terminal",
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Capture a single raw frame from a ZED camera through zed-sdk-mlx."""

    _ensure_zed_capture_constraints(width=width, height=height, framerate=framerate)
    resolved_repo = resolve_zed_sdk_mlx_repo(repo_path)
    if resolved_repo is None:
        raise MacCameraCaptureError(
            f"zed-sdk-mlx capture requires --zed-sdk-mlx-repo or ${ZED_SDK_MLX_ENV} to point at the checkout"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    app_bin = _build_zed_capture_app(resolved_repo)

    if launch_mode == "direct":
        completed = _run_completed_command(
            [str(app_bin), "--device", str(device_index), str(output_file)],
            cwd=resolved_repo,
            timeout_s=timeout_s,
        )
        detail = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if completed.returncode != 0:
            raise MacCameraCaptureError(_format_zed_capture_failure(detail, source="zed-sdk-mlx direct capture"))
    elif launch_mode == "terminal":
        with tempfile.TemporaryDirectory(prefix="isaaclab-zed-capture-") as tmp_dir:
            temp_dir = Path(tmp_dir)
            log_path = temp_dir / "zed_capture_terminal.log"
            status_path = temp_dir / "zed_capture_terminal.status"
            script_path = _write_terminal_capture_script(
                repo_path=resolved_repo,
                app_bin=app_bin,
                output_path=output_file,
                log_path=log_path,
                status_path=status_path,
                device_index=device_index,
            )
            _open_terminal_script(script_path)
            _wait_for_terminal_capture(
                output_path=output_file,
                status_path=status_path,
                log_path=log_path,
                timeout_s=timeout_s,
            )
    else:
        raise MacCameraCaptureError(f"unsupported zed-sdk-mlx launch mode: {launch_mode}")

    metadata_path = Path(f"{output_file}.txt")
    metadata = validate_raw_capture_artifacts(output_file, metadata_path)
    return {
        "backend": f"zed-sdk-mlx-{launch_mode}",
        "repo_path": str(resolved_repo),
        "output_path": str(output_file),
        "metadata_path": str(metadata_path),
        "device_index": device_index,
        "width": metadata.get("width", width),
        "height": metadata.get("height", height),
        "pixel_format": "yuyv422",
        "timestamp": metadata.get("timestamp"),
        "serial_number": metadata.get("serial_number"),
        "frame_id": metadata.get("frame_id"),
    }


def capture_mac_camera_raw_frame(
    *,
    device_index: int,
    output_path: str | Path,
    width: int,
    height: int,
    framerate: int | float = 30,
    ffmpeg_bin: str = "ffmpeg",
    timeout_s: float = 60.0,
    pixel_format: str = "yuyv422",
    capture_backend: str = CAPTURE_BACKEND_AUTO,
    zed_sdk_mlx_repo: str | Path | None = None,
    device: MacCameraDevice | None = None,
) -> dict[str, Any]:
    """Capture one raw frame through the requested backend, auto-selecting ZED flow when available."""

    if capture_backend not in CAPTURE_BACKENDS:
        raise MacCameraCaptureError(
            f"unsupported capture backend {capture_backend!r}; expected one of {', '.join(CAPTURE_BACKENDS)}"
        )

    resolved_backend = capture_backend
    resolved_zed_repo = resolve_zed_sdk_mlx_repo(zed_sdk_mlx_repo)
    if resolved_backend == CAPTURE_BACKEND_AUTO:
        if device is not None and device.is_stereolabs and resolved_zed_repo is not None:
            resolved_backend = CAPTURE_BACKEND_ZED_TERMINAL
        else:
            resolved_backend = CAPTURE_BACKEND_AVFOUNDATION

    if resolved_backend == CAPTURE_BACKEND_AVFOUNDATION:
        payload = capture_avfoundation_raw_frame(
            device_index=device_index,
            output_path=output_path,
            width=width,
            height=height,
            framerate=framerate,
            ffmpeg_bin=ffmpeg_bin,
            timeout_s=timeout_s,
            pixel_format=pixel_format,
        )
        payload["backend"] = CAPTURE_BACKEND_AVFOUNDATION
        return payload

    if resolved_backend == CAPTURE_BACKEND_ZED_DIRECT:
        return capture_zed_sdk_mlx_raw_frame(
            device_index=device_index,
            output_path=output_path,
            width=width,
            height=height,
            framerate=framerate,
            repo_path=resolved_zed_repo,
            launch_mode="direct",
            timeout_s=timeout_s,
        )

    if resolved_backend == CAPTURE_BACKEND_ZED_TERMINAL:
        return capture_zed_sdk_mlx_raw_frame(
            device_index=device_index,
            output_path=output_path,
            width=width,
            height=height,
            framerate=framerate,
            repo_path=resolved_zed_repo,
            launch_mode="terminal",
            timeout_s=timeout_s,
        )

    raise MacCameraCaptureError(f"unsupported capture backend resolution: {resolved_backend!r}")
