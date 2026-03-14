# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.backends.mac_sim import parse_avfoundation_device_list, parse_avfoundation_supported_modes


def test_parse_avfoundation_device_list_filters_screens():
    output = """
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] ZED 2i
[AVFoundation indev @ 0x1] [1] MacBook Pro Camera
[AVFoundation indev @ 0x1] [2] Capture screen 0
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
