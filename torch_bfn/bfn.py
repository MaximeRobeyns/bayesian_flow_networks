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

from torch_bfn.utils import str_to_torch_dtype


class ContinuousBFN(nn.Module):
    def __init__(
        self,
        dim: int,
        net: nn.Module,
        *,
        device_str: str = "cpu",
        dtype_str: str = "float32",
        eps: float = 1e-9,
    ):
        """Continuous-time Bayesian Flow Network

        Args:
            dim: The number of data dimensions
            net: The network to use; mapping [B, D] x [B] -> [B, D]
            device_str: PyTorch device to use
            dtype_str: PyTorch dtype to use
            eps: stability parameter
        """
        super().__init__()
        self.device = t.device(device_str)
        self.dtype = str_to_torch_dtype(dtype_str)
        self.dim = dim

        dtype_eps = t.finfo(self.dtype).eps
        self.eps = eps if eps < dtype_eps else dtype_eps

        self.net = net.to(self.device, self.dtype)
        self.net.train()

        # Assert that the network has the right dimensions
        test_batch = t.randn((16, dim), device=self.device, dtype=self.dtype)
        test_time = t.rand((16, 1), device=self.device, dtype=self.dtype)
        out = self.net(test_batch, test_time)
        assert out.shape == (16, dim)

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
        zeros = t.zeros_like(mu)
        eps = self.net(mu, time)
        x = (mu / gamma) - t.sqrt((1.0 - gamma) / gamma) * eps
        x = t.clip(x, x_min, x_max)
        return t.where(time < t_min, zeros, x)

    def loss(self, x: Tensor["B", "D"], sigma_1: float = 0.002) -> Tensor["B"]:
        """Continuous-time loss function; L∞(t)

        Args:
            x: training data
            sigma_1: standard deviation at t=1.

        Returns:
            Tensor["B"]: batch loss
        """
        s1 = t.tensor([sigma_1], device=x.device, dtype=self.dtype)
        time = t.rand((*x.shape[:-1], 1), device=x.device, dtype=self.dtype)
        gamma = 1.0 - s1.pow(2.0 * time)
        dist = t.distributions.Normal(gamma * x, gamma * (1 - gamma) + self.eps)
        mu = dist.sample((1,)).squeeze(0)
        x_pred = self.cts_output_prediction(mu, time, gamma)
        loss = -(s1.log() * (x - x_pred).pow(2.0) / s1.pow(2 * time)).mean(-1)
        return loss

    def discrete_loss(
        self, x: Tensor["B", "D"], sigma_1: float = 0.002, n: int = 30
    ) -> Tensor["B"]:
        """Discrete (n-step) loss function for continuous data.

        Args:
            x: training data
            sigma_1: standard deviatoin at t=1
            n: number of training steps

        Returns:
            Tensor["B"]: batch loss
        """
        s1 = t.tensor([sigma_1], device=x.device)
        i = t.randint(1, n + 1, (*x.shape[:-1], 1)).to(x.device)
        time = (i - 1) / n
        gamma = 1.0 - s1.pow(2.0 * time)
        mask = (gamma != 0).squeeze(-1)
        mu = t.zeros_like(x)
        gnz = gamma[mask]
        dist = t.distributions.Normal(gnz * x[mask], gnz * (1 - gnz))
        mu[mask] = dist.sample((1,)).squeeze(0)
        x_pred = t.zeros_like(mu)
        x_pred[mask] = self.cts_output_prediction(
            mu[mask], time[mask], gamma[mask]
        )
        loss = (n * (1.0 - s1.pow(2.0 / n))) / (2.0 * s1.pow(2.0 * i / n))
        loss = loss * (x - x_pred).pow(2.0)
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
            y_dist = t.distributions.Normal(x, 1 / alpha + self.eps)
            y = y_dist.sample((1,)).squeeze(0)
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho = rho + alpha
        return self.cts_output_prediction(
            mu,
            t.tensor((1,), device=self.device).expand(n_samples, 1),
            1 - s1.pow(2.0),
        )
