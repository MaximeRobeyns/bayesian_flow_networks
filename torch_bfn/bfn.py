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
import torch.nn.functional as F

from typing import Tuple, Optional, Union
from torchtyping import TensorType as Tensor
from torch.distributions import Categorical, Normal

from torch_bfn.utils import str_to_torch_dtype, exists, default
from torch_bfn.networks import BFNetwork, DiscreteBFNetwork


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
        """
        Bayesian flow network for continuous data.

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

    def _pad_to_dim(self, a: Tensor["B"]) -> Tensor["B", "1*dim"]:
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
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> Tensor["B"]:
        """Continuous-time loss function; L∞(t)

        Args:
            x: training data
            cond: an optional class / conditioning vector
            sigma_1: standard deviation at t=1
            cond_scale: scale to apply to the conditional signal
            rescaled_phi: another lever to influence the conditioning signal

        Returns:
            Tensor["B"]: batch loss
        """
        s1 = t.tensor([sigma_1], device=x.device, dtype=self.dtype)
        time = t.rand((x.size(0),), device=x.device, dtype=self.dtype)
        time = self._pad_to_dim(time)
        gamma = 1.0 - s1.pow(2.0 * time)
        std = (gamma * (1 - gamma) + self.eps).sqrt()
        mu = gamma * x + std * t.randn_like(x)
        x_pred = self.cts_output_prediction(
            mu, time, gamma, cond, cond_scale, rescaled_phi
        )
        diff = (x - x_pred).flatten(1).pow(2.0).sum(-1).sqrt()
        loss = -(s1.log() * diff / s1.pow(2 * time.view(-1)))
        return loss

    def discrete_loss(
        self,
        x: Tensor["B", "D"],
        cond: Optional[Tensor["B", "C"]] = None,
        sigma_1: float = 0.002,
        n: int = 30,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> Tensor["B"]:
        """Discrete (n-step) loss function for continuous data.

        Args:
            x: training data
            cond: conditioning / class information
            sigma_1: standard deviatoin at t=1
            n: number of training steps
            cond_scale: scale to apply to the conditional signal
            rescaled_phi: another lever to influence the conditioning signal

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
        std = (gnz * (1 - gnz)).sqrt()
        mu[mask] = gnz * x[mask] + std * t.randn_like(x[mask])
        x_pred = t.zeros_like(mu)
        cts_output = self.cts_output_prediction(
            mu[mask],
            time[mask],
            gamma[mask],
            cond[mask],
            cond_scale,
            rescaled_phi,
        )
        x_pred[mask] = cts_output
        loss = (n * (1.0 - s1.pow(2.0 / n))) / (2.0 * s1.pow(2.0 * i / n))
        diff = (x - x_pred).flatten(1).pow(2.0).sum(-1).sqrt()
        loss = loss * diff
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
            gamma = 1 - s1.pow(2 * time)
            x = self.cts_output_prediction(
                mu, time, gamma, cond, cond_scale, rescaled_phi
            )
            alpha = s1.pow(-2 * i / n_timesteps) * (1 - s1.pow(2 / n_timesteps))
            std = (1 / alpha + self.eps).sqrt()
            y = x + std * t.randn_like(x)
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


class DiscreteBFN(nn.Module):
    """
    A discrete Bayesian flow network, where every dimension of your problem
    must have the same number of class labels, K.

    This is unsuitable, for instance, for Bayesian optimisation over
    categorical variables, where each categorical variable may have different
    K.

    This however comes to the benefit of simplicity and performance for
    problems where each dimension does have the same number of class labels,
    such as language modelling, discrete images (where each pixel has the same
    dimemension) and so forth.
    """

    def __init__(
        self,
        dim: Union[Tuple[int], int],
        K: int,
        net: DiscreteBFNetwork,
        *,
        beta_1: float = 0.2,
        device_str: str = "cpu",
        dtype_str: str = "float32",
        eps: float = 1e-9,
    ):
        """
        Bayesian flow network for discrete data.
        """
        super().__init__()
        self.device = t.device(device_str)
        self.dtype = str_to_torch_dtype(dtype_str)
        self.dim = dim if isinstance(dim, Tuple) else (dim,)
        self.K = K
        assert beta_1 > 0.0, "beta_1 must be positive"
        self.beta_1 = beta_1

        dtype_eps = t.finfo(self.dtype).eps
        self.eps = eps if eps < dtype_eps else dtype_eps

        self.net = net.to(self.device, self.dtype)
        self.net.train()

        # Assert that the network has the right dimensions
        bs = 16
        test_batch = t.randn(
            (bs, *self.dim, K), device=self.device, dtype=self.dtype
        )
        test_time = t.rand((bs,), device=self.device, dtype=self.dtype)
        # We use the presence of cond_dim as a way to identify conditional models
        if net.is_conditional_model:
            classes = t.randint(0, net.cond_dim, (bs, 1), device=self.device)
        else:
            classes = None
        out = self.net(test_batch, test_time, classes)
        assert out.shape == (bs, *self.dim, self.K)

    def _pad_to_dim(self, a: Tensor["B"]) -> Tensor["B", "1*dim"]:
        return a.view(a.shape[0], *((1,) * len(self.dim)))

    def discrete_output_probs(
        self,
        theta: Tensor["B", "D", "K", bool],
        time: Tensor["B"],
        cond: Optional[Tensor["B", "C"]] = None,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> Tensor["B", "D", "K"]:
        """Returns the probs (not distribution) of the output distribution."""
        assert (time >= 0).all() and (time <= 1).all()
        # assert theta
        if cond is not None:
            if exists(cond_scale) or exists(rescaled_phi):
                psi = self.net.forward_with_cond_scale(
                    theta,
                    time.view(-1),
                    cond,
                    cond_scale=default(cond_scale, 1.0),
                    rescaled_phi=default(rescaled_phi, 0.0),
                )
            else:
                psi = self.net(theta, time.view(-1), cond)
        else:
            psi = self.net(theta, time.view(-1))
        assert psi.shape == theta.shape
        if psi.size(-1) == 1:
            # Handle Bernoulli parameters separately
            p1 = t.sigmoid(psi)
            probs = t.cat((p1, 1 - p1), -1)
        else:
            probs = t.softmax(psi, -1)
        assert t.allclose(probs.sum(-1), t.ones_like(probs[..., 0]))
        return probs

    def loss(
        self,
        x: Tensor["B", "D", int],
        beta_1: Optional[float] = None,
        cond: Optional[Tensor["B", "C"]] = None,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> Tensor["B"]:
        """Continuous-time loss for discrete data"""
        beta_1 = default(beta_1, self.beta_1)
        assert beta_1 > 0.0
        assert (x >= 0).all() and (
            x < self.K
        ).all(), f"x must contain class labels between 0 and {self.K}"

        time = t.rand((x.size(0),), device=x.device, dtype=self.dtype)
        beta = beta_1 * time.pow(2.0)
        e_x = F.one_hot(x, num_classes=self.K).float()
        mean = beta.view(-1, 1, 1) * (self.K * e_x - 1)
        std = (beta * self.K).view(-1, 1, 1).sqrt()
        y_samples = mean + std * t.randn_like(mean)
        theta = t.softmax(y_samples, -1)
        out_probs = self.discrete_output_probs(
            theta, time, cond, cond_scale, rescaled_phi
        )
        diff = (e_x - out_probs).pow(2.0)
        loss = self.K * beta_1 * time[:, None, None] * diff
        return loss

    def discrete_loss(
        self,
        x: Tensor["B", "D", int],
        cond: Optional[Tensor["B", "C"]] = None,
        beta_1: Optional[float] = None,
        n: int = 30,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> Tensor["B"]:
        """Discrete-time loss for discrete data

        WARNING: not getting good results with this one. Might have a bug.
        Consider using the continuous-time `loss` instead.
        """
        beta_1 = default(beta_1, self.beta_1)
        assert beta_1 > 0.0
        assert (x >= 0).all() and (
            x < self.K
        ).all(), f"x must contain class labels between 0 and {self.K}"

        i = t.randint(1, n + 1, (x.size(0),)).to(x.device)
        time = (i - 1) / n
        beta = beta_1 * time.pow(2.0)
        e_x = F.one_hot(x, num_classes=self.K).float()
        mean = beta.view(-1, 1, 1) * (self.K * e_x - 1)
        std = (beta * self.K).view(-1, 1, 1).sqrt()
        yp_samples = mean + std * t.randn_like(mean)
        theta = t.softmax(yp_samples, -1)
        out_probs = self.discrete_output_probs(
            theta, time, cond, cond_scale, rescaled_phi
        )
        alpha = beta_1 * (2 * i - 1) / n**2
        y_mean = alpha.view(-1, 1, 1) * (self.K * e_x - 1)
        y_std = (alpha * self.K).view(-1, 1, 1).sqrt()
        y_dist = Normal(y_mean, y_std)
        y_samples = y_dist.sample((1,)).squeeze(0)
        y_lp = y_dist.log_prob(y_samples)
        l1 = y_lp.sum(-1).sum(-1)
        l2 = (out_probs * y_lp.exp()).sum(-1).log().sum(-1)
        loss = n * (l1 - l2)
        return loss

    @t.inference_mode()
    def sample(
        self,
        n_samples: int = 10,
        beta_1: Optional[float] = None,
        n_timesteps: int = 20,
        cond: Optional[Tensor["Y", "C"]] = None,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> Union[Tensor["n_samples", "dim"], Tensor["n_samples", "Y", "dim"]]:
        if exists(cond):
            if cond.ndim == 1:
                cond = cond[:, None]
            n_cond = cond.size(0)
            cond = cond.repeat_interleave(n_samples, 0)

        self.net.eval()
        tkwargs = {"device": self.device, "dtype": self.dtype}
        b1 = t.tensor((default(beta_1, self.beta_1),), **tkwargs)
        theta = t.full((n_samples, *self.dim, self.K), 1 / self.K, **tkwargs)
        for i in range(1, n_timesteps + 1):
            time = t.tensor(((i - 1) / n_timesteps,), **tkwargs)
            time = self._pad_to_dim(time)
            out_probs = self.discrete_output_probs(
                theta, time, cond, cond_scale, rescaled_phi
            )
            out_dist = Categorical(probs=out_probs)
            k_samples = out_dist.sample((1,)).squeeze(0)
            alpha = b1 * ((2 * i - 1) / n_timesteps**2)
            e_k = F.one_hot(k_samples, num_classes=self.K).float()
            std = (alpha * self.K).sqrt()
            y = alpha * (self.K * e_k - 1) + std * t.randn_like(std)
            theta_prime = y.exp() * theta
            theta = t.softmax(theta_prime, -1)
        out_probs = self.discrete_output_probs(
            theta, t.tensor((1,), **tkwargs), cond, cond_scale, rescaled_phi
        )
        samples = Categorical(out_probs).sample((1,)).squeeze(0)
        self.net.train()
        if exists(cond):
            samples = samples.view(n_cond, n_samples, *samples.shape[1:])
        return samples
