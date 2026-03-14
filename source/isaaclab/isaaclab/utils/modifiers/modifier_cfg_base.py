# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, TypeAlias

from isaaclab.utils import configclass

if TYPE_CHECKING:
    import torch

    TensorLike: TypeAlias = torch.Tensor
else:
    TensorLike: TypeAlias = Any


@configclass
class ModifierCfg:
    """Configuration parameters for stateless modifiers."""

    func: Callable[..., TensorLike] = MISSING
    """Function or callable class used by modifier."""

    params: dict[str, Any] = dict()
    """The parameters to be passed to the function or callable class as keyword arguments."""
