# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in mac-sim tests.
    torch = None
try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - exercised in mac-sim tests.
    wp = None


def configure_seed(seed: int | None, torch_deterministic: bool = False) -> int:
    """Set seed across all random number generators (torch, numpy, random, warp).

    Args:
        seed: The random seed value. If None, generates a random seed.
        torch_deterministic: If True, enables deterministic mode for torch operations.

    Returns:
        The seed value that was set.
    """
    if seed is None or seed == -1:
        seed = 42 if torch_deterministic else random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    if wp is not None:
        wp.rand_init(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        if torch is not None and torch.cuda.is_available():
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if torch is not None:
            torch.use_deterministic_algorithms(True)
    else:
        if torch is not None and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    return seed
