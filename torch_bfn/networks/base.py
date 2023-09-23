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
"""Base network classes for torch_bfn, as well as some utilities"""


import math
import torch as t
import torch.nn as nn

from abc import abstractmethod
from typing import Optional
from functools import partial
from torchtyping import TensorType as Tensor

__all__ = [
    "BFNetwork",
    "DiscreteBFNetwork",
    "SinusoidalPosEmb",
    "RandomOrLearnedSinusoidalPosEmb",
]


class BFNetwork(nn.Module):
    """
    Abstraact base class for neural networks (nn.Module) for use with
    torch_bfn.
    """

    def __init__(self, is_conditional_model: bool = False):
        super().__init__()
        self.is_conditional_model = is_conditional_model

    @abstractmethod
    def forward(
        self,
        x: Tensor["B", "D"],
        time: Tensor["B"],
        cond: Optional[Tensor["B", "C"]] = None,
        cond_drop_prob: Optional[float] = None,
    ) -> Tensor["B", "D"]:
        """Returns a value of the same shape as x (e.g. predicts the noise
        applied to x) at time t with optional conditioning information.

        Args:
            x: the current parameter vector
            time: current timestep
            cond: conditioning information
            cond_drop_prob: probability of dropping conditioning info out for
                classifier-free guidance.

        Returns:
            Tensor["B", "D"]: updated parameter vector
        """
        raise NotImplementedError

    def forward_with_cond_scale(
        self, *args, cond_scale=1.0, rescaled_phi=0.0, **kwargs
    ) -> Tensor["B", "D"]:
        """For conditional sampling, this additionally scales the conditional
        guidance, and sharpens phi.

        This abstract class just invokes the forward method as a fallback.
        """
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)
        if cond_scale == 1.0:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.0:
            return scaled_logits

        std_fn = partial(
            t.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True
        )
        rescaled_logits = scaled_logits * (
            std_fn(logits) / std_fn(scaled_logits)
        )
        return rescaled_logits * rescaled_phi + scaled_logits * (
            1.0 - rescaled_phi
        )


class DiscreteBFNetwork(BFNetwork):
    """
    For discrete variants of BFNs, we require networks to use the last tensor
    dimemsion as the class label dimension, as is conventional in transformer
    models for language.
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor["B", "D", "K"],
        time: Tensor["B"],
        cond: Optional[Tensor["B", "C"]],
        cond_drop_prob: Optional[float] = None,
    ) -> Tensor["B", "D", "K"]:
        raise NotImplementedError


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = t.exp(t.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = t.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        assert dim % 2 == 0, "Sinusoidal positional embedding dim must be even"
        half_dim = dim // 2
        self.weights = nn.Parameter(
            t.randn(half_dim), requires_grad=not is_random
        )

    def forward(self, x: Tensor["B", 1]) -> Tensor["B", "dim+1"]:
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = t.cat((freqs.sin(), freqs.cos()), -1)
        fouriered = t.cat((x, fouriered), -1)
        return fouriered
