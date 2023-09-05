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
"""
Linear residual networks for use in simple settings.
"""

import torch as t
import torch.nn as nn

from typing import Optional
from torchtyping import TensorType as Tensor

from torch_bfn.utils import default, exists
from torch_bfn.networks.base import (
    BFNetwork,
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
)

__all__ = ["LinearNetwork", "DiscreteLinearNet"]


class LinearBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, x: Tensor["B", "in_dim"], scale_shift=None
    ) -> Tensor["B", "out_dim"]:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.dropout(x)
        return x


class LinearResnetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_emb_dim: int = 16,
        cond_emb_dim: Optional[int] = None,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        mlp_dim = default(time_emb_dim, 0) + default(cond_emb_dim, 0)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mlp_dim, out_dim * 2),
        )
        self.block1 = LinearBlock(in_dim, out_dim, dropout_p)
        self.block2 = LinearBlock(out_dim, out_dim, dropout_p)
        self.res_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: Tensor["B", "in_dim"],
        time_emb: Optional[Tensor["B", "time_emb_dim"]] = None,
        cond_emb: Optional[Tensor["B", "cond_emb_dim"]] = None,
    ) -> Tensor["B", "out_dim"]:
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(cond_emb)):
            emb = tuple(filter(exists, (time_emb, cond_emb)))
            emb = self.mlp(t.cat(emb, -1))
            scale_shift = emb.chunk(2, dim=-1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_proj(x)


class LinearNetwork(BFNetwork):
    def __init__(
        self,
        dim: int = 2,
        hidden_dims: list[int] = [128, 128],
        cond_dim: Optional[int] = None,
        cond_drop_prob: float = 0.5,
        sin_dim: int = 16,
        time_dim: int = 16,
        random_time_emb: bool = False,
        dropout_p: float = 0.0,
    ):
        """Simple network for D-dimensional data.

        Args:
            dim: data dimension
            hidden_dims: Hidden features to use in the network
            cond_dim: dimension of conditioning information
            cond_drop_prob: probability of dropping conditioning info out
                during classifier-free guidance
            sin_dim: simusoidal time embedding dims
            time_dim: time embedding dimension
            random_time_emb: use random (True) or learned (False) time embedding
            dropout_p: dropout used in network
        """
        super().__init__(cond_dim)

        # Time embeddings
        self.time_mlp = nn.Sequential(
            RandomOrLearnedSinusoidalPosEmb(sin_dim, random_time_emb),
            nn.Linear(sin_dim + 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embeddings
        if self.is_conditional_model:
            self.cond_dim = cond_dim
            self.cond_drop_prob = cond_drop_prob
            self.cond_emb = nn.Embedding(self.cond_dim, dim)
            self.null_classes_emb = nn.Parameter(t.randn(dim))

            cond_embed_dim = dim * 4

            self.cond_mlp = nn.Sequential(
                nn.Linear(dim, cond_embed_dim),
                nn.GELU(),
                nn.Linear(cond_embed_dim, cond_embed_dim),
            )
        else:
            cond_embed_dim = None
            self.cond_emb = None
            self.cond_drop_prob = 1.0

        hs = [dim] + hidden_dims
        self.blocks = nn.ModuleList([])
        for j, k in zip(hs[:-1], hs[1:]):
            self.blocks.append(
                LinearResnetBlock(j, k, time_dim, cond_embed_dim, dropout_p)
            )
        self.final_proj = nn.Linear(hs[-1], dim)

    def forward(
        self,
        x: Tensor["B", "D"],
        time: Tensor["B"],
        cond: Optional[Tensor["B"]] = None,
        cond_drop_prob: Optional[float] = None,
    ) -> Tensor["B", "D"]:
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if time.shape == (1,):
            time = time.expand(batch)
        time = self.time_mlp(time[:, None])

        # Handle conditioning information

        if self.is_conditional_model:
            if not exists(cond):
                cond = t.randint(0, self.cond_dim, (batch,), device=device)
                cond_drop_prob = 1.0

            # Recover from cond of shape [B, 1] instead of [B]
            if cond.ndim > 1:
                if cond.ndim == 2 and cond.size(-1) == 1:
                    cond = cond.squeeze(-1)
                else:
                    raise ValueError(
                        f"Class shape should be ({batch},), not {cond.shape}"
                    )

            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
            cond_emb = self.cond_emb(cond)

            if cond_drop_prob > 0.0:
                keep_mask = t.rand((batch,), device=device) < (
                    1 - cond_drop_prob
                )
                null_classes_emb = self.null_classes_emb.expand(batch, -1)
                cond_emb = t.where(
                    keep_mask[:, None], cond_emb, null_classes_emb
                )
            c = self.cond_mlp(cond_emb)
        else:
            c = None

        x_res = x.clone()
        for block in self.blocks:
            x = block(x, time, c)
        x = self.final_proj(x)
        return x + x_res


class DiscreteLinearNet(LinearNetwork):
    def __init__(
        self,
        dim: int = 2,
        K: int = 1,
        hidden_dims: list[int] = [128, 128],
        cond_dim: Optional[int] = None,
        cond_drop_prob: float = 0.5,
        sin_dim: int = 16,
        time_dim: int = 16,
        random_time_emb: bool = False,
        dropout_p: float = 0.0,
    ):
        """Simple network for D-dimensional, discrete data.

        Args:
            dim: data dimension
            K: number of classes per dimension
            hidden_dims: Hidden features to use in the network
            cond_dim: dimension of conditioning information
            cond_drop_prob: probability of dropping conditioning info out
                during classifier-free guidance
            sin_dim: simusoidal time embedding dims
            time_dim: time embedding dimension
            random_time_emb: use random (True) or learned (False) time embedding
            dropout_p: dropout used in network
        """
        LinearNetwork.__init__(
            self,
            dim * K,
            hidden_dims,
            cond_dim,
            cond_drop_prob,
            sin_dim,
            time_dim,
            random_time_emb,
            dropout_p,
        )

    def forward(
        self,
        x: Tensor["B", "D", "K"],
        time: Tensor["B"],
        cond: Optional[Tensor["B", "C"]] = None,
        cond_drop_prob: Optional[float] = None,
    ) -> Tensor["B", "D", "K"]:
        return (
            super()
            .forward(x.flatten(1), time, cond, cond_drop_prob)
            .view(x.shape)
        )
