# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing different noise models implementations.

The noise models are implemented as functions that take in a tensor and a configuration and return a tensor
with the noise applied. These functions are then used in the :class:`NoiseCfg` configuration class.

Usage:

.. code-block:: python

    import torch
    from isaaclab.utils.noise import AdditiveGaussianNoiseCfg

    # create a random tensor
    my_tensor = torch.rand(128, 128, device="cuda")

    # create a noise configuration
    cfg = AdditiveGaussianNoiseCfg(mean=0.0, std=1.0)

    # apply the noise
    my_noisified_tensor = cfg.func(my_tensor, cfg)

"""
from __future__ import annotations

import importlib

_MODULE_EXPORTS = {
    "NoiseCfg": (".noise_cfg", "NoiseCfg"),
    "ConstantNoiseCfg": (".noise_cfg", "ConstantNoiseCfg"),
    "GaussianNoiseCfg": (".noise_cfg", "GaussianNoiseCfg"),
    "NoiseModelCfg": (".noise_cfg", "NoiseModelCfg"),
    "NoiseModelWithAdditiveBiasCfg": (".noise_cfg", "NoiseModelWithAdditiveBiasCfg"),
    "UniformNoiseCfg": (".noise_cfg", "UniformNoiseCfg"),
    "NoiseModel": (".noise_model", "NoiseModel"),
    "NoiseModelWithAdditiveBias": (".noise_model", "NoiseModelWithAdditiveBias"),
    "constant_noise": (".noise_model", "constant_noise"),
    "gaussian_noise": (".noise_model", "gaussian_noise"),
    "uniform_noise": (".noise_model", "uniform_noise"),
}

__all__ = [*_MODULE_EXPORTS.keys(), "ConstantBiasNoiseCfg", "AdditiveUniformNoiseCfg", "AdditiveGaussianNoiseCfg"]


def __getattr__(name: str):
    if name == "ConstantBiasNoiseCfg":
        value = __getattr__("ConstantNoiseCfg")
    elif name == "AdditiveUniformNoiseCfg":
        value = __getattr__("UniformNoiseCfg")
    elif name == "AdditiveGaussianNoiseCfg":
        value = __getattr__("GaussianNoiseCfg")
    else:
        target = _MODULE_EXPORTS.get(name)
        if target is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])

    globals()[name] = value
    return value

# Backward compatibility
