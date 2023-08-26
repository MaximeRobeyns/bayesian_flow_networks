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
"""Main BFN methods"""

import torch as t
import torch.nn as nn

from torchtyping import TensorType as Tensor

from torch_bfn.networks import (
    RandomOrLearnedSinusoidalPosEmb,
    LinearResnetBlock,
)
from torch_bfn.utils import str_to_torch_dtype


class BFNNetwork(nn.Module):
    """A very simple network to use for BFN testing"""

    def __init__(
        self,
        dim: int,
        hidden_dims: list[int],
        sin_dim: int,
        time_dim: int,
        random_time_emb: bool,
        dropout_p: float,
    ):
        super().__init__()
        # self.time_mlp = nn.Sequential(
        #     RandomOrLearnedSinusoidalPosEmb(sin_dim, random_time_emb),
        #     nn.Linear(sin_dim + 1, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim),
        # )
        self.time_mlp = RandomOrLearnedSinusoidalPosEmb(
            sin_dim, random_time_emb
        )
        time_dim = sin_dim + 1

        hs = [dim] + hidden_dims
        self.blocks = nn.ModuleList([])
        for j, k in zip(hs[:-1], hs[1:]):
            self.blocks.append(LinearResnetBlock(j, k, time_dim, dropout_p))
        self.final_proj = nn.Linear(hs[-1], dim)

    def forward(
        self, x: Tensor["B", "D"], time: Tensor["B"]
    ) -> Tensor["B", "D"]:
        time = self.time_mlp(time)
        for block in self.blocks:
            x = block(x, time)
        x = self.final_proj(x)
        return x


class BFN(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        hidden_dims: list[int] = [128, 128],
        sin_dim: int = 16,
        time_dim: int = 16,
        random_time_emb: bool = False,
        dropout_p: float = 0.0,
        device_str: str = "cuda:0",
        dtype_str: str = "float64",
    ):
        super().__init__()
        self.device = t.device(device_str)
        self.dtype = str_to_torch_dtype(dtype_str)
        self.dim = dim

        self.net = BFNNetwork(
            dim, hidden_dims, sin_dim, time_dim, random_time_emb, dropout_p
        ).to(self.device, self.dtype)
        self.net.train()

    def cts_output_prediction(
        self,
        mu: Tensor["B", "D"],
        time: Tensor["B", 1],
        gamma: Tensor["B", 1],
        t_min=1e-10,
        x_min=-1.0,
        x_max=1.0,
    ) -> Tensor["B", "D"]:
        assert (time >= 0).all() and (time <= 1).all()
        assert mu.dim() == time.dim()
        eps = self.net(mu, time)
        x = (mu / gamma) - t.sqrt((1.0 - gamma) / gamma) * eps
        x = t.clip(x, x_min, x_max)
        return t.where(time < t_min, 1e-5 * eps, x)

    def loss(self, x: Tensor["B", "D"], sigma_1: float = 0.002) -> Tensor["B"]:
        """
        Continuous-time loss function for continuous data.
        """
        s1 = t.tensor([sigma_1], device=x.device)
        time = t.rand((*x.shape[:-1], 1), device=x.device)
        gamma = 1.0 - s1.pow(2.0 * time)
        # mu = gamma * x + gamma * (1 - gamma) * t.randn_like(x)
        # mu = gamma * x + t.sqrt(gamma * (1 - gamma)) * t.randn_like(x)
        dist = t.distributions.Normal(gamma * x, gamma * (1 - gamma))
        mu = dist.sample((1,)).squeeze(0)
        x_pred = self.cts_output_prediction(mu, time, gamma)
        loss = -(s1.log() * (x - x_pred).pow(2.0) / s1.pow(2 * time)).mean(-1)
        return loss

    def discrete_loss(
        self, x: Tensor["B", "D"], sigma_1: float = 0.002, n: int = 30
    ) -> Tensor["B"]:
        s1 = t.tensor([sigma_1], device=x.device)
        i = t.randint(1, n + 1, (*x.shape[:-1], 1)).to(x.device)
        time = (i - 1) / n
        gamma = 1.0 - s1.pow(2.0 * time)
        mu = gamma * x + gamma * (1 - gamma) * t.randn_like(x)
        x_pred = self.cts_output_prediction(mu, time, gamma)
        loss = (n * (1.0 - s1.pow(2.0 / n))) / (2.0 * s1.pow(2.0 * i / n))
        loss *= (x - x_pred).pow(2.0)
        return loss

    @t.inference_mode()
    def sample(
        self, n_samples: int = 10, sigma_1: float = 0.001, n_timesteps: int = 20
    ) -> Tensor["n_samples", "dim"]:
        self.net.eval()
        s1 = t.tensor((sigma_1,), device=self.device, dtype=self.dtype)
        mu = t.zeros(
            (n_samples, self.dim), device=self.device, dtype=self.dtype
        )
        rho = 1.0
        for i in range(1, n_timesteps + 1):
            time = t.tensor(
                ((i - 1) / n_timesteps,), device=self.device, dtype=self.dtype
            )
            time = time.expand(n_samples, 1)
            gamma = 1 - s1.pow(2 * time)
            x = self.cts_output_prediction(mu, time, gamma)
            alpha = s1.pow(-2 * i / n_timesteps) * (1 - s1.pow(2 / n_timesteps))
            y_dist = t.distributions.Normal(x, 1 / alpha)
            y = y_dist.sample((1,)).squeeze(0)
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho = rho + alpha
        return self.cts_output_prediction(
            mu,
            t.tensor((1,), device=self.device).expand(n_samples, 1),
            1 - s1.pow(2.0),
        )
