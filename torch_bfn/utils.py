# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions"""
from __future__ import annotations

import copy
import numpy as np
import torch as t
import torch.nn as nn

from typing import Any, Callable, Dict, Optional, TypeVar, Tuple, Union
from functools import wraps
from torchtyping import TensorType as Tensor

T = TypeVar("T")


def copy_mod(mod: nn.Module) -> nn.Module:
    new_mod = copy.copy(mod)
    new_mod._parameters = copy.copy(mod._parameters)
    new_mod._buffers = copy.copy(mod._buffers)
    new_mod._modules = {
        name: copy_mod(child) for (name, child) in mod._modules.items()
    }
    return new_mod


def get_fst_device(model: nn.Module) -> t.device:
    """Returns the device of the first layer in the given module"""
    return next(model.parameters()).device


class EMA(object):
    """
    Maintains an exponential moving average of the registered module's
    parameters.
    """

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow: Dict[str, t.Tensor] = {}

    def register(self, module: nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module: nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module: nn.Module) -> nn.Module:
        module_copy = copy_mod(module)
        # module_copy = type(module)(module.config).to(module.config.device)
        # module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> Dict[str, t.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, t.Tensor]):
        self.shadow = state_dict


def norm_denorm(
    data: Tensor["B", "D"], xmin: float = -1.0, xmax: float = 1.0
) -> Tuple[Tensor["B", "D"], Callable[[Tensor["B", "D"]], Tensor["B", "D"]]]:
    """Normalises data to the given range, and returns a denormalising function.

    Args:
        data: The data to normalise; batch and dim must be 1D, respectively.
        xmin: minimum range for all dimensions
        xmax: maximum range for all dimensions

    Returns:
        normalised data and denormalising function
    """
    # Normalise to min=0, max=1
    mins = data.min(0).values
    maxs = data.max(0).values
    ranges = maxs - mins
    data = (data - mins) / ranges
    # Shift to target range
    data = data * (xmax - xmin) + xmin

    def denorm(x: Tensor["B", "D"]) -> Tensor["B", "D"]:
        x = (x - xmin) / (xmax - xmin)
        return x * ranges + mins

    return data, denorm


def exists(x: Optional[Any]) -> bool:
    return x is not None


def default(val: Optional[T], d: Union[T, Callable[[], T]]) -> T:
    if val is not None:
        return val
    return d() if callable(d) else d


def cast_tuple(t: Union[Tuple[T, ...], T], length: int = 1) -> Tuple[T, ...]:
    if isinstance(t, tuple):
        return t
    return (t,) * length


def once(fn: Callable[[Any], Any]):
    called = False

    @wraps(fn)
    def inner(x: Any):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def str_to_torch_dtype(name: str) -> t.dtype:
    dt = t.__dict__[name]
    assert isinstance(dt, t.dtype)
    return dt


def np_to_torch_dtype(np_type: str | np.dtype) -> t.dtype:
    match np_type:
        case "bool" | np.bool_:
            return t.bool
        case "uint8" | np.uint8:
            return t.uint8
        case "int8" | np.int8:
            return t.int8
        case "int16" | np.int16:
            return t.int16
        case "int32" | np.int32:
            return t.int32
        case "int64" | np.int64:
            return t.int64
        case "float16" | np.float16:
            return t.float16
        case "float32" | np.float32:
            return t.float32
        case "float64" | np.float64:
            return t.float64
        case "complex64" | np.complex64:
            return t.complex64
        case "complex128" | np.complex128:
            return t.complex128
        case _:
            raise ValueError(f"Unrecognized type, {np_type}")


# Some random fun functions which might be useful:


class Squareplus(nn.Module):
    # https://arxiv.org/pdf/2112.11687.pdf

    def __init__(self, a=2):
        super().__init__()
        self.a = a

    def forward(self, x: Tensor[...]) -> Tensor[...]:
        """The 'squareplus' activation function: has very similar properties to
        softplus, but is far cheaper computationally.
            - squareplus(0) = 1 (softplus(0) = ln 2)
            - gradient diminishes more slowly for negative inputs.
            - ReLU = (x + sqrt(x^2))/2
            - 'squareplus' becomes smoother with higher 'a'
        """
        return (x + t.sqrt(t.square(x) + self.a * self.a)) / 2


def squareplus_f(x: Tensor[...], a: int = 2) -> Tensor[...]:
    """The 'squareplus' activation function: has very similar properties to
    softplus, but is far cheaper computationally.
        - squareplus(0) = 1 (softplus(0) = ln 2)
        - gradient diminishes more slowly for negative inputs.
        - ReLU = (x + sqrt(x^2))/2
        - 'squareplus' becomes smoother with higher 'a'
    """
    return (x + t.sqrt(t.square(x) + a * a)) / 2


def symlog(x: Tensor[...]) -> Tensor[...]:
    """Useful as a stateless normalisation function.
        - leaves well-normalised values alone (approximates identity around 0)
        - compresses both large positive and negative values
        - symmetric around origin + preserves input sign (unlike logarithm)

    https://arxiv.org/pdf/2301.04104.pdf
    """
    return t.sign(x) * t.log(t.abs(x) + 1)


def symexp(y: Tensor[...]) -> Tensor[...]:
    """Inverse of symlog; use to denormalise network predictions when trained
    on symlog-normalised data.
        - allows network predictions to quickly move to large values if needed
    """
    return t.sign(y) * (t.exp(t.abs(y)) - 1)
