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

from typing import Tuple, Optional, Union
from torchtyping import TensorType as Tensor

from torch_bfn.utils import str_to_torch_dtype, exists, default
from torch_bfn.networks import BFNetwork


class ContinuousBFN(nn.Module):
    def __init__(
        self,
        dim: Union[Tuple[int], int],
        net: BFNetwork,
        *,
        device_str: str = "cpu",
        dtype_str: str = "float32",
        eps: float = 1e-9,
    ):
        """Continuous-time Bayesian Flow Network

        Args:
            dim: The dimensions of the data e.g. (8,) for 8-dimensional
                vectors, or (3, 64, 64) for RGB images with 64x64 pixels.
            net: The network to use; mapping [B, D] x [B] -> [B, D]
            device_str: PyTorch device to use
            dtype_str: PyTorch dtype to use
            eps: stability parameter
        """
        super().__init__()
        self.device = t.device(device_str)
        self.dtype = str_to_torch_dtype(dtype_str)
        self.dim = dim if isinstance(dim, Tuple) else (dim,)

        dtype_eps = t.finfo(self.dtype).eps
        self.eps = eps if eps < dtype_eps else dtype_eps

        self.net = net.to(self.device, self.dtype)
        self.net.train()

        # Assert that the network has the right dimensions
        bs = 16
        test_batch = t.randn(
            (bs, *self.dim), device=self.device, dtype=self.dtype
        )
        test_time = t.rand((bs,), device=self.device, dtype=self.dtype)
        # We use the presence of cond_dim as a way to identify conditional models
        if net.is_conditional_model:
            classes = t.randint(0, net.cond_dim, (bs, 1), device=self.device)
        else:
            classes = None
        out = self.net(test_batch, test_time, classes)
        assert out.shape == (bs, *self.dim)

    def _pad_to_dim(self, a: Tensor["B"]):
        return a.view(a.shape[0], *((1,) * len(self.dim)))

    def cts_output_prediction(
        self,
        mu: Tensor["B", "D"],
        time: Tensor["B", 1],
        gamma: Tensor["B", 1],
        cond: Optional[Tensor["B", "C"]] = None,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
        t_min=1e-10,
        x_min=-1.0,
        x_max=1.0,
    ) -> Tensor["B", "D"]:
        assert (time >= 0).all() and (time <= 1).all()
        assert mu.dim() == time.dim()
        zeros = t.zeros_like(mu)
        if cond is not None:
            if exists(cond_scale) or exists(rescaled_phi):
                eps = self.net.forward_with_cond_scale(
                    mu,
                    time.view(-1),
                    cond,
                    cond_scale=default(cond_scale, 1.0),
                    rescaled_phi=default(rescaled_phi, 0.0),
                )
            else:
                eps = self.net(mu, time.view(-1), cond)
        else:
            eps = self.net(mu, time.view(-1))
        x = (mu / gamma) - t.sqrt((1.0 - gamma) / gamma) * eps
        x = t.clip(x, x_min, x_max)
        return t.where(time < t_min, zeros, x)

    def loss(
        self,
        x: Tensor["B", "D"],
        cond: Optional[Tensor["B", "C"]] = None,
        sigma_1: float = 0.002,
    ) -> Tensor["B"]:
        """Continuous-time loss function; L∞(t)

        Args:
            x: training data
            cond: optional class / conditioning information
            sigma_1: standard deviation at t=1.

        Returns:
            Tensor["B"]: batch loss
        """
        s1 = t.tensor([sigma_1], device=x.device, dtype=self.dtype)
        time = t.rand((x.size(0),), device=x.device, dtype=self.dtype)
        time = self._pad_to_dim(time)
        gamma = 1.0 - s1.pow(2.0 * time)
        dist = t.distributions.Normal(
            gamma * x, (gamma * (1 - gamma) + self.eps).sqrt()
        )
        mu = dist.sample((1,)).squeeze(0)
        x_pred = self.cts_output_prediction(mu, time, gamma, cond)
        loss = -(s1.log() * (x - x_pred).pow(2.0) / s1.pow(2 * time)).mean(-1)
        return loss

    def discrete_loss(
        self,
        x: Tensor["B", "D"],
        cond: Tensor["B", "C"],
        sigma_1: float = 0.002,
        n: int = 30,
    ) -> Tensor["B"]:
        """Discrete (n-step) loss function for continuous data.

        Args:
            x: training data
            cond: conditioning / class information
            sigma_1: standard deviatoin at t=1
            n: number of training steps

        Returns:
            Tensor["B"]: batch loss
        """
        s1 = t.tensor([sigma_1], device=x.device)
        i = t.randint(1, n + 1, (x.size(0),)).to(x.device)
        i = self._pad_to_dim(i)
        time = (i - 1) / n
        gamma = 1.0 - s1.pow(2.0 * time)
        mask = gamma.view(-1) != 0
        mu = t.zeros_like(x)
        gnz = gamma[mask]  # gamma non-zero
        dist = t.distributions.Normal(gnz * x[mask], (gnz * (1 - gnz)).sqrt())
        mu[mask] = dist.sample((1,)).squeeze(0)
        x_pred = t.zeros_like(mu)
        cts_output = self.cts_output_prediction(
            mu[mask], time[mask], gamma[mask], cond[mask]
        )
        x_pred[mask] = cts_output
        loss = (n * (1.0 - s1.pow(2.0 / n))) / (2.0 * s1.pow(2.0 * i / n))
        loss = loss * (x - x_pred).pow(2.0)
        return loss

    @t.inference_mode()
    def sample(
        self,
        n_samples: int = 10,
        sigma_1: float = 0.001,
        n_timesteps: int = 20,
        cond: Optional[Tensor["Y", "cond_dim"]] = None,
        cond_scale: Optional[float] = None,  # 1.
        rescaled_phi: Optional[float] = None,  # 0.0,
    ) -> Union[Tensor["n_samples", "dim"], Tensor["n_samples", "Y", "dim"]]:
        if exists(cond):
            if cond.ndim == 1:
                cond = cond[:, None]
            n_cond = cond.size(0)
            cond = cond.repeat_interleave(n_samples, 0)
            batch = cond.size(0)
        else:
            batch = n_samples

        self.net.eval()
        tkwargs = {"device": self.device, "dtype": self.dtype}
        s1 = t.tensor((sigma_1,), **tkwargs)
        mu = t.zeros((batch, *self.dim), **tkwargs)
        rho = 1.0
        for i in range(1, n_timesteps + 1):
            time = t.tensor(((i - 1) / n_timesteps,), **tkwargs)
            time = self._pad_to_dim(time)
            # time = time.expand(cond.size(0))
            gamma = 1 - s1.pow(2 * time)
            x = self.cts_output_prediction(
                mu, time, gamma, cond, cond_scale, rescaled_phi
            )
            alpha = s1.pow(-2 * i / n_timesteps) * (1 - s1.pow(2 / n_timesteps))
            y_dist = t.distributions.Normal(x, (1 / alpha + self.eps).sqrt())
            y = y_dist.sample((1,)).squeeze(0)
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho = rho + alpha
        t1 = self._pad_to_dim(t.tensor((1,), **tkwargs))
        outputs = self.cts_output_prediction(
            mu, t1, 1 - s1.pow(2.0), cond, cond_scale, rescaled_phi
        )
        self.net.train()
        if cond is not None:
            outputs = outputs.view(n_cond, n_samples, *outputs.shape[1:])
        return outputs
