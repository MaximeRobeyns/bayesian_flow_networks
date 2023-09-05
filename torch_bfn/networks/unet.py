# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
# Copyright (C) 2020 Phil Wang <github.com/lucidrains>
# Copyright (C) 2020 Jonathan Ho <github.com/hojonathanho>
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
"""Unets, usually used with image data [C, W, H].

References:
https://github.com/hojonathanho/diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Any, Callable, Optional, Tuple, Union
from functools import partial
from packaging import version
from collections import namedtuple
from torchtyping import TensorType as Tensor

from torch_bfn.utils import default, exists, cast_tuple, print_once
from torch_bfn.networks.base import (
    BFNetwork,
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
)

__all__ = ["Unet"]


class Residual(nn.Module):
    def __init__(self, fn: Callable[[Tensor["D":...], Any], Tensor["D":...]]):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor["D":...], *args, **kwargs) -> Tensor["D":...]:
        return self.fn(x, *args, **kwargs) + x


def upsample(dim: int, out_dim: Optional[int] = None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(out_dim, dim), 3, padding=1),
    )


def downsample(dim: int, out_dim: Optional[int] = None) -> nn.Module:
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(out_dim, dim), 1),
    )


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, groups: int = 8):
        super().__init__()
        # assert (
        #     out_dim % groups == 0
        # ), f"groups ({groups}) does not divide out dim ({out_dim}) in Block"
        groups = 1 if out_dim % groups != 0 else groups
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_dim)
        self.act = nn.SiLU()

    def forward(
        self, x: t.Tensor, scale_shift: Optional[Tuple[float, float]] = None
    ) -> t.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        time_emb_dim: Optional[int] = None,
        class_emb_dim: Optional[int] = None,
        groups: int = 8,
    ):
        super().__init__()
        mlp_dim = default(time_emb_dim, 0) + default(class_emb_dim, 0)
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(mlp_dim, out_dim * 2))
            if exists(time_emb_dim) or exists(class_emb_dim)
            else None
        )
        self.block1 = Block(in_dim, out_dim, groups=groups)
        self.block2 = Block(out_dim, out_dim, groups=groups)
        self.res_conv = (
            nn.Conv2d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor["B", "in_dim"],
        time_emb: Optional[Tensor["B", "time_emb_dim"]] = None,
        class_emb: Optional[Tensor["B", "class_emb_dim"]] = None,
    ) -> Tensor["B", "out_dim"]:
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            emb = tuple(filter(exists, (time_emb, class_emb)))
            emb = self.mlp(t.cat(emb, -1))[..., None, None]
            scale_shift = emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(t.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


AttentionConfig = namedtuple(
    "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


class Attend(nn.Module):
    def __init__(self, dropout: float = 0.0, flash: bool = False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(t.__version__) < version.parse("2.0.0")
        ), "Flash Attention requires torch>=2.0"

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not t.cuda.is_available() or not flash:
            return

        device_properties = t.cuda.get_device_properties(t.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 detected; using flash attention if input is on CUDA"
            )
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once(
                "No A100 GPU detected; using math / mem efficient attention if input on CUDA"
            )
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q: t.Tensor, k: t.Tensor, v: t.Tensor) -> t.Tensor:
        q, k, v = map(lambda a: a.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if q.is_cuda else self.cpu_config
        with t.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        return out

    def forward(self, q: t.Tensor, k: t.Tensor, v: t.Tensor) -> t.Tensor:
        # Note on einsum notation:
        # b = batch
        # h = heads
        # n, i, j = seq lengths (base, source, target)
        # d = feature dimension

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5

        # Similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # Attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Aggregate values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1), RMSNorm(dim)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = t.einsum("b h d n, b h e n -> b h d e", k, v)
        out = t.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w
        )
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 4, dim_head=32, flash: bool = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: t.Tensor) -> t.Tensor:
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda a: rearrange(a, "b (h c) x y -> b h (x y) c", h=self.heads),
            qkv,
        )

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(BFNetwork):
    def __init__(
        self,
        dim: int,
        dim_mults: list[int] = [1, 2, 2],
        channels: int = 3,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        cond_drop_prob: float = 0.5,
        resnet_block_groups: int = 8,
        learned_sinusoidal_cond: bool = False,
        learned_sinusoidal_dim: int = 16,
        random_fourier_features: bool = False,
        full_attn: Union[Tuple[bool, ...], bool] = (False, False, True),
        attn_heads: Union[Tuple[int, ...], int] = 4,
        attn_dim_head: Union[Tuple[int, ...], int] = 32,
        flash_attn: bool = False,
    ):
        super().__init__(num_classes)

        # Set up dimensions
        self.channels = channels
        self.init_dim = default(init_dim, dim)
        self.out_dim = default(out_dim, channels)
        self.init_conv = nn.Conv2d(channels, self.init_dim, 7, padding=3)

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embeddings
        if self.is_conditional_model:
            self.cond_dim = num_classes
            self.cond_drop_prob = cond_drop_prob
            self.class_emb = nn.Embedding(self.cond_dim, dim)
            self.null_classes_emb = nn.Parameter(t.randn(dim))

            classes_dim = dim * 4

            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim),
            )
        else:
            classes_dim = None
            self.class_emb = None
            self.cond_drop_prob = 1.0

        # Attention
        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        block_class = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            time_emb_dim=time_dim,
            class_emb_dim=classes_dim,
        )

        for i, (
            (in_dim, out_dim),
            full_attn_i,
            attn_heads_i,
            attn_dim_head_i,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = i >= (num_resolutions - 1)

            attn_class = FullAttention if full_attn_i else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(in_dim, in_dim),
                        block_class(in_dim, in_dim),
                        attn_class(
                            in_dim,
                            dim_head=attn_dim_head_i,
                            heads=attn_heads_i,
                        ),
                        downsample(in_dim, out_dim)
                        if not is_last
                        else nn.Conv2d(in_dim, out_dim, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim)
        self.mid_attn = FullAttention(
            mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
        )
        self.mid_block2 = block_class(mid_dim, mid_dim)

        for i, (
            (in_dim, out_dim),
            full_attn_i,
            attn_heads_i,
            attn_dim_head_i,
        ) in enumerate(
            zip(
                *(map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
            )
        ):
            is_last = i == (len(in_out) - 1)
            attn_class = FullAttention if full_attn_i else LinearAttention
            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(out_dim + in_dim, out_dim),
                        block_class(out_dim + in_dim, out_dim),
                        attn_class(
                            out_dim,
                            dim_head=attn_dim_head_i,
                            heads=attn_heads_i,
                        ),
                        upsample(out_dim, in_dim)
                        if not is_last
                        else nn.Conv2d(out_dim, in_dim, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_class(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.downs) - 1)

    def forward(
        self,
        x: Tensor["B", "D"],
        time: Tensor["B"],
        classes: Optional[Tensor["B", 1]] = None,
        cond_drop_prob: Optional[float] = None,
    ) -> Tensor["B", "D"]:
        batch, device = x.shape[0], x.device

        # Handle conditoning information

        if self.is_conditional_model:

            if not exists(classes):
                classes = t.randint(0, self.cond_dim, (batch,), device=device)
                cond_drop_prob = 1.0

            # Recover from class of shape [B, 1] instead of [B]
            if classes.ndim > 1:
                if classes.ndim == 2 and classes.size(-1) == 1:
                    classes = classes.squeeze(-1)
                else:
                    raise ValueError(
                        f"Class shape should be ({batch},), not {classes.shape}"
                    )

            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
            classes_emb = self.class_emb(classes)

            if cond_drop_prob > 0.0:
                keep_mask = t.rand((batch,), device=device) < (
                    1 - cond_drop_prob
                )
                null_classes_emb = self.null_classes_emb.expand(batch, -1)
                classes_emb = t.where(
                    keep_mask[:, None], classes_emb, null_classes_emb
                )

            c = self.classes_mlp(classes_emb)
        else:
            c = None

        # main unet

        assert all(
            [d % self.downsample_factor == 0 for d in x.shape[-2:]]
        ), f"input dim {x.shape[-2:]} must be divisible by {self.downsample_factor} for Unet"

        x = self.init_conv(x)
        r = x.clone()

        if time.shape == (1,):
            time = time.expand(batch)
        time = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time, c)
            h.append(x)

            x = block2(x, time, c)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, time, c)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time, c)

        for block1, block2, attn, upsample in self.ups:
            x = t.cat((x, h.pop()), 1)
            x = block1(x, time, c)

            x = t.cat((x, h.pop()), 1)
            x = block2(x, time, c)
            x = attn(x) + x

            x = upsample(x)

        x = t.cat((x, r), dim=1)
        x = self.final_res_block(x, time, c)
        return self.final_conv(x)
