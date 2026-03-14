# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Retargeters for mapping input-device data to robot commands."""

from __future__ import annotations

import importlib

_MODULE_EXPORTS = {
    "GR1T2Retargeter": (".humanoid.fourier.gr1t2_retargeter", "GR1T2Retargeter"),
    "GR1T2RetargeterCfg": (".humanoid.fourier.gr1t2_retargeter", "GR1T2RetargeterCfg"),
    "G1LowerBodyStandingRetargeter": (
        ".humanoid.unitree.g1_lower_body_standing",
        "G1LowerBodyStandingRetargeter",
    ),
    "G1LowerBodyStandingRetargeterCfg": (
        ".humanoid.unitree.g1_lower_body_standing",
        "G1LowerBodyStandingRetargeterCfg",
    ),
    "G1LowerBodyStandingMotionControllerRetargeter": (
        ".humanoid.unitree.g1_motion_controller_locomotion",
        "G1LowerBodyStandingMotionControllerRetargeter",
    ),
    "G1LowerBodyStandingMotionControllerRetargeterCfg": (
        ".humanoid.unitree.g1_motion_controller_locomotion",
        "G1LowerBodyStandingMotionControllerRetargeterCfg",
    ),
    "UnitreeG1Retargeter": (".humanoid.unitree.inspire.g1_upper_body_retargeter", "UnitreeG1Retargeter"),
    "UnitreeG1RetargeterCfg": (".humanoid.unitree.inspire.g1_upper_body_retargeter", "UnitreeG1RetargeterCfg"),
    "G1TriHandUpperBodyMotionControllerGripperRetargeter": (
        ".humanoid.unitree.trihand.g1_upper_body_motion_ctrl_gripper",
        "G1TriHandUpperBodyMotionControllerGripperRetargeter",
    ),
    "G1TriHandUpperBodyMotionControllerGripperRetargeterCfg": (
        ".humanoid.unitree.trihand.g1_upper_body_motion_ctrl_gripper",
        "G1TriHandUpperBodyMotionControllerGripperRetargeterCfg",
    ),
    "G1TriHandUpperBodyMotionControllerRetargeter": (
        ".humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter",
        "G1TriHandUpperBodyMotionControllerRetargeter",
    ),
    "G1TriHandUpperBodyMotionControllerRetargeterCfg": (
        ".humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter",
        "G1TriHandUpperBodyMotionControllerRetargeterCfg",
    ),
    "G1TriHandUpperBodyRetargeter": (
        ".humanoid.unitree.trihand.g1_upper_body_retargeter",
        "G1TriHandUpperBodyRetargeter",
    ),
    "G1TriHandUpperBodyRetargeterCfg": (
        ".humanoid.unitree.trihand.g1_upper_body_retargeter",
        "G1TriHandUpperBodyRetargeterCfg",
    ),
    "GripperRetargeter": (".manipulator.gripper_retargeter", "GripperRetargeter"),
    "GripperRetargeterCfg": (".manipulator.gripper_retargeter", "GripperRetargeterCfg"),
    "Se3AbsRetargeter": (".manipulator.se3_abs_retargeter", "Se3AbsRetargeter"),
    "Se3AbsRetargeterCfg": (".manipulator.se3_abs_retargeter", "Se3AbsRetargeterCfg"),
    "Se3RelRetargeter": (".manipulator.se3_rel_retargeter", "Se3RelRetargeter"),
    "Se3RelRetargeterCfg": (".manipulator.se3_rel_retargeter", "Se3RelRetargeterCfg"),
}

__all__ = list(_MODULE_EXPORTS.keys())


def __getattr__(name: str):
    target = _MODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
