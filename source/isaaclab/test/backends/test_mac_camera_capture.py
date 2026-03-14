# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

import isaaclab.backends.mac_sim.cameras as cameras_module
from isaaclab.backends.mac_sim import (
    MacCameraCaptureError,
    MacCameraDevice,
    capture_mac_camera_raw_frame,
    parse_avfoundation_device_list,
    parse_avfoundation_supported_modes,
    parse_capture_metadata_text,
    resolve_zed_sdk_mlx_repo,
    validate_raw_capture_artifacts,
)


def test_parse_avfoundation_device_list_filters_screens():
    output = """
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] ZED 2i
[AVFoundation indev @ 0x1] [1] MacBook Pro Camera
[AVFoundation indev @ 0x1] [2] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
""".strip()

    devices = parse_avfoundation_device_list(output)

    assert [device.index for device in devices] == [0, 1]
    assert devices[0].name == "ZED 2i"
    assert devices[0].is_stereolabs is True
    assert devices[1].is_stereolabs is False


def test_parse_avfoundation_supported_modes_deduplicates_entries():
    output = """
[avfoundation @ 0x1] Supported modes:
[avfoundation @ 0x1]   1344x376@[30.000030 30.000030]fps
[avfoundation @ 0x1]   1344x376@[30.000030 30.000030]fps
[avfoundation @ 0x1]   2560x720@[15.000015 30.000030]fps
""".strip()

    modes = parse_avfoundation_supported_modes(output)

    assert [(mode.width, mode.height) for mode in modes] == [(1344, 376), (2560, 720)]
    assert modes[0].min_fps == 30.00003
    assert modes[1].max_fps == 30.00003


def test_parse_capture_metadata_text_coerces_numeric_fields():
    metadata = parse_capture_metadata_text(
        "\n".join(
            [
                "width=2560",
                "height=720",
                "channels=2",
                "frame_id=7",
                "timestamp=123456789",
                "serial_number=38892829",
                "framerate=30.0",
                "pixel_format=yuyv422",
            ]
        )
    )

    assert metadata["width"] == 2560
    assert metadata["height"] == 720
    assert metadata["channels"] == 2
    assert metadata["frame_id"] == 7
    assert metadata["timestamp"] == 123456789
    assert metadata["serial_number"] == 38892829
    assert metadata["framerate"] == 30.0
    assert metadata["pixel_format"] == "yuyv422"


def test_resolve_zed_sdk_mlx_repo_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    (tmp_path / "Makefile").write_text("build-capture-app:\n\t@true\n", encoding="utf-8")
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "zed_capture_app.mm").write_text("// smoke\n", encoding="utf-8")
    monkeypatch.setenv("ZED_SDK_MLX_REPO", str(tmp_path))

    resolved = resolve_zed_sdk_mlx_repo()

    assert resolved == tmp_path.resolve()


def test_capture_mac_camera_raw_frame_auto_selects_zed_terminal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[tuple[str, int]] = []

    def fake_zed_capture(**kwargs):
        calls.append((kwargs["launch_mode"], kwargs["device_index"]))
        return {"backend": "zed-sdk-mlx-terminal", "output_path": str(kwargs["output_path"])}

    monkeypatch.setattr(cameras_module, "resolve_zed_sdk_mlx_repo", lambda repo_path=None: tmp_path)
    monkeypatch.setattr(cameras_module, "capture_zed_sdk_mlx_raw_frame", fake_zed_capture)

    payload = capture_mac_camera_raw_frame(
        device_index=0,
        output_path=tmp_path / "frame.yuv",
        width=2560,
        height=720,
        capture_backend="auto",
        zed_sdk_mlx_repo=tmp_path,
        device=MacCameraDevice(index=0, name="ZED 2i"),
    )

    assert payload["backend"] == "zed-sdk-mlx-terminal"
    assert calls == [("terminal", 0)]


def test_capture_mac_camera_raw_frame_auto_falls_back_to_avfoundation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[int] = []

    def fake_av_capture(**kwargs):
        calls.append(kwargs["device_index"])
        return {"backend": "avfoundation", "output_path": str(kwargs["output_path"])}

    monkeypatch.setattr(cameras_module, "resolve_zed_sdk_mlx_repo", lambda repo_path=None: None)
    monkeypatch.setattr(cameras_module, "capture_avfoundation_raw_frame", fake_av_capture)

    payload = capture_mac_camera_raw_frame(
        device_index=1,
        output_path=tmp_path / "frame.yuv",
        width=640,
        height=480,
        capture_backend="auto",
        device=MacCameraDevice(index=1, name="MacBook Pro Camera"),
    )

    assert payload["backend"] == "avfoundation"
    assert calls == [1]


def test_capture_mac_camera_raw_frame_rejects_unsupported_zed_resolution(tmp_path: Path):
    (tmp_path / "Makefile").write_text("build-capture-app:\n\t@true\n", encoding="utf-8")
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "zed_capture_app.mm").write_text("// smoke\n", encoding="utf-8")

    with pytest.raises(MacCameraCaptureError, match="2560x720"):
        capture_mac_camera_raw_frame(
            device_index=0,
            output_path=tmp_path / "frame.yuv",
            width=1920,
            height=1080,
            capture_backend="zed-sdk-mlx-terminal",
            zed_sdk_mlx_repo=tmp_path,
        )


def test_validate_raw_capture_artifacts_rejects_size_mismatch(tmp_path: Path):
    output_path = tmp_path / "frame.yuv"
    output_path.write_bytes(b"\x00" * 12)
    metadata_path = tmp_path / "frame.yuv.txt"
    metadata_path.write_text("width=4\nheight=4\nchannels=2\ntimestamp=1\n", encoding="utf-8")

    with pytest.raises(MacCameraCaptureError, match="size does not match metadata"):
        validate_raw_capture_artifacts(output_path, metadata_path)
