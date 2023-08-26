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
"""Comparison diffusion model"""

__all__ = ["DDPM"]

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import Optional
from torchtyping import TensorType as Tensor

from torch_bfn.utils import str_to_torch_dtype


class NoiseSchedule(Enum):
    linear = "linear"
    cosine = "cosine"
    sigmoid = "sigmoid"


class ConditionalLinear(nn.Module):
    def __init__(self, num_in: int, num_out: int, num_time_embeddings: int):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(num_time_embeddings, num_out)
        self.embed.weight.data.uniform_()

    def forward(
        self, x: Tensor["B", "num_in"], time: Tensor["B", int]
    ) -> Tensor["B", "num_out"]:
        x = x.to(self.lin.weight.data.dtype)
        out = self.lin(x)
        gamma = self.embed(time)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps: int):
        super().__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = nn.Linear(128, 2)

    def forward(
        self, x: Tensor["B", "D"], time: Tensor["B"]
    ) -> Tensor["B", "D"]:
        x = F.softplus(self.lin1(x, time))
        x = F.softplus(self.lin2(x, time))
        return self.lin3(x)


class DDPM(nn.Module):
    def __init__(
        self,
        n_steps: int = 100,
        schedule: str = "sigmoid",
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
        device_str: str = "cuda:0",
        dtype_str: str = "float64",
    ):
        if schedule not in NoiseSchedule.__members__:
            raise ValueError(
                f"Unrecognised noise schedule: {schedule}"
                f"Choose from {NoiseSchedule.__members__}"
            )
        super().__init__()
        self.device = t.device(device_str)
        self.dtype = str_to_torch_dtype(dtype_str)
        self.n_steps = n_steps

        self.betas = self._make_beta_schedule(
            NoiseSchedule(schedule), n_steps, beta_start, beta_end
        ).to(self.device, self.dtype)

        self.alphas = 1 - self.betas
        alphas_prod = t.cumprod(self.alphas, 0)
        ones = t.tensor([1], dtype=self.dtype, device=self.device)
        self.alphas_prod_p = t.cat((ones, alphas_prod[:-1]), 0)
        self.alphas_bar_sqrt = t.sqrt(alphas_prod)
        self.one_minus_alphas_bar_log = t.log(1 - alphas_prod)
        self.one_minus_alphas_bar_sqrt = t.sqrt(1 - alphas_prod)

        self.model = ConditionalModel(self.n_steps)
        self.model = self.model.to(self.device, self.dtype)

    def _make_beta_schedule(
        self,
        schedule: NoiseSchedule,
        n_timesteps: int,
        start: float,
        end: float,
    ) -> Tensor["n_timesteps"]:
        if schedule == schedule.linear:
            betas = t.linspace(start, end, n_timesteps)
        elif schedule == schedule.cosine:
            betas = t.linspace(start**0.5, end**0.5, n_timesteps) ** 2
        elif schedule == schedule.sigmoid:
            betas = t.linspace(-6, 6, n_timesteps)
            betas = t.sigmoid(betas) * (end - start) + start
        return betas

    @staticmethod
    def _extract(
        inp: Tensor["A"], time: Tensor["N"], x: Tensor["B", "D"]
    ) -> Tensor["N", "ones"]:
        shape = x.shape
        out = t.gather(inp, 0, time.to(inp.device))
        reshape = [time.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    @t.inference_mode()
    def q_sample(
        self,
        x_0: Tensor["B", "D"],
        time: Tensor["N"],
        noise: Optional[Tensor["B", "D"]] = None,
    ) -> Tensor["B", "D"]:
        if noise is None:
            noise = t.randn_like(x_0)
        alphas_t = self._extract(self.alphas_bar_sqrt, time, x_0)
        alphas_1_m_t = self._extract(self.one_minus_alphas_bar_sqrt, time, x_0)
        return alphas_t * x_0 + alphas_1_m_t * noise

    @t.inference_mode()
    def p_sample(self, x: Tensor["B", "D"], time: int) -> Tensor["B", "D"]:
        time = t.tensor([time]).to(x.device)  # type: ignore
        eps_factor = (1 - self._extract(self.alphas, time, x)) / self._extract(
            self.one_minus_alphas_bar_sqrt, time, x
        )
        eps_theta = self.model(x, time)
        mean = (1 / self._extract(self.alphas, time, x).sqrt()) * (
            x - (eps_factor * eps_theta)
        )
        z = t.randn_like(x)
        sigma_t = self._extract(self.betas, time, x).sqrt()
        sample = mean + sigma_t * z
        return sample

    @t.inference_mode()
    def p_sample_loop(self, shape: t.Size) -> list[Tensor["B", "D"]]:
        x_seq = [t.randn(shape, device=self.device, dtype=self.dtype)]
        for i in reversed(range(self.n_steps)):
            x_seq.append(self.p_sample(x_seq[-1], i))
        x_seq = [x.cpu() for x in x_seq]
        return x_seq

    def noise_estimation_loss(self, x_0: Tensor["B", "D"]) -> Tensor[1]:
        batch_size = x_0.shape[0]
        # Select a random step for each example
        time = t.randint(
            0, self.n_steps, size=(batch_size // 2 + 1,), device=x_0.device
        )
        time = t.cat([time, self.n_steps - time - 1], dim=0)[:batch_size].long()
        a = self._extract(self.alphas_bar_sqrt, time, x_0)
        # eps multiplier
        am1 = self._extract(self.one_minus_alphas_bar_sqrt, time, x_0)
        e = t.randn_like(x_0)
        # model input
        x = x_0 * a + e * am1
        output = self.model(x, time)
        return (e - output).square().mean()
