# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for working with different array backends."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations

from typing import Union

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in mac-sim tests.
    torch = None

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - exercised in mac-sim tests.
    wp = None

TensorData = Union[np.ndarray]  # noqa: UP007
"""Type definition for a tensor data.

Union of numpy, torch, and warp arrays.
"""

TENSOR_TYPES = {
    "numpy": np.ndarray,
}
"""A dictionary containing the types for each backend.

The keys are the name of the backend ("numpy", "torch", "warp") and the values are the corresponding type
(``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""

TENSOR_TYPE_CONVERSIONS = {
    "numpy": {},
}
"""A nested dictionary containing the conversion functions for each backend.

The keys of the outer dictionary are the name of target backend ("numpy", "torch", "warp"). The keys of the
inner dictionary are the source backend (``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""

if wp is not None:
    TensorData = Union[np.ndarray, wp.array]  # noqa: UP007
    TENSOR_TYPES["warp"] = wp.array
    TENSOR_TYPE_CONVERSIONS["numpy"][wp.array] = lambda x: x.numpy()
    TENSOR_TYPE_CONVERSIONS["warp"] = {np.ndarray: lambda x: wp.array(x)}

if torch is not None:
    TensorData = Union[np.ndarray, torch.Tensor]  # noqa: UP007
    TENSOR_TYPES["torch"] = torch.Tensor
    TENSOR_TYPE_CONVERSIONS["numpy"][torch.Tensor] = lambda x: x.detach().cpu().numpy()
    TENSOR_TYPE_CONVERSIONS["torch"] = {np.ndarray: lambda x: torch.from_numpy(x)}
    if wp is not None:
        TensorData = Union[np.ndarray, torch.Tensor, wp.array]  # noqa: UP007
        TENSOR_TYPE_CONVERSIONS["torch"][wp.array] = lambda x: wp.torch.to_torch(x)
        TENSOR_TYPE_CONVERSIONS["warp"][torch.Tensor] = lambda x: wp.torch.from_torch(x)


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for `convert_to_torch`, but it is not installed.")

    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    elif wp is not None and isinstance(array, wp.array):
        if array.dtype == wp.uint32:
            array = array.view(wp.int32)
        tensor = wp.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor
