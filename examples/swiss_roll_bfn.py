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
"""Swiss roll example using a diffusion model"""

import torch as t
import matplotlib.pyplot as plt

from typing import Callable, Tuple
from torchtyping import TensorType as Tensor
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch_bfn import BFN
from torch_bfn.utils import EMA, get_fst_device, norm_denorm, str_to_torch_dtype


def make_roll_dset(
    n: int, bs: int = 128, noise: float = 0.3, dtype: t.dtype = t.float64
) -> Tuple[
    DataLoader, DataLoader, Callable[[Tensor["B", "D"]], Tensor["B", "D"]]
]:
    # Create a normalised 'swiss roll' dataset
    X_np, _ = make_swiss_roll(n_samples=n, noise=noise)
    X_np = X_np[:, [0, 2]] / 10.0
    X = t.tensor(X_np, dtype=dtype)
    X, denorm = norm_denorm(X)
    dset = TensorDataset(X)
    train_size = len(dset) - bs
    val_size = len(dset) - train_size
    train_dset, val_dset = random_split(dset, [train_size, val_size])
    train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=bs, shuffle=False)
    return train_loader, val_loader, denorm


def plot_samples(
    denormed_samples: Tensor["B", "D"], fpath: str = "outputs/samples.png"
):
    samples = denormed_samples.numpy()
    plt.figure(figsize=(6, 4))
    plt.scatter(samples[:, 0], samples[:, 1], edgecolor="k")
    plt.title("Posterior Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(fpath)
    plt.close()


@t.inference_mode()
def plot_trajectories(
    model: BFN,
    n_samples: int = 10,
    sigma_1: float = 0.002,
    n: int = 100,
    eps=1e-8,
):
    model.eval()
    device = get_fst_device(model)
    xs = t.linspace(-3.0, 3.0, 200).to(device)
    s1 = t.tensor([sigma_1]).to(device)
    mus, probs = [], []
    mu = t.zeros((n_samples, 1)).to(device)
    mus.append(mu[None, :].cpu().clone().detach())
    probs.append(
        t.distributions.Normal(0, 1)
        .log_prob(xs)[None, :]
        .cpu()
        .clone()
        .detach()
    )
    rho = 1.0
    for i in range(1, n + 1):
        time = t.tensor([[(i - 1) / n]]).to(device).expand(n_samples, 1)
        gamma = 1 - s1.pow(2 * time)
        x = model.cts_output_prediction(mu, time, gamma)
        alpha = s1.pow(-2 * i / n) * (1 - s1.pow(2 / n))
        y = x + t.randn_like(x) / alpha
        mu = (rho * mu + alpha * y) / (rho + alpha)
        mus.append(mu[None, :].cpu())
        # mus.append((y)[None, :].cpu())
        bfdist = t.distributions.Normal(
            gamma * x[:1], gamma * (1 - gamma) + eps
        )
        probs.append(bfdist.log_prob(xs).mean(0, keepdim=True).detach().cpu())
        rho = rho + alpha
    samples = (
        model.cts_output_prediction(
            mu, t.tensor([[1.0]]).to(0).expand(n_samples, 1), 1 - s1.pow(2)
        ).cpu(),
    )
    return samples, mus, probs


def train(
    model: BFN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    denorm: Callable[[Tensor["B", "D"]], Tensor["B", "D"]],
    epochs: int = 2001,
    device_str: str = "cuda:0",
    dtype_str: str = "float64",
):
    device = t.device(device_str)
    dtype = str_to_torch_dtype(dtype_str)
    ema = EMA(0.9)

    model.to(device, dtype)
    opt = t.optim.Adam(model.parameters(), lr=1e-3)
    ema.register(model)

    for epoch in range(epochs):
        loss = None
        for batch in train_loader:
            X = batch[0].to(device, dtype)
            loss = model.loss(X, sigma_1=0.01).mean()
            opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        if epoch % 200 == 0:
            assert loss is not None
            print(loss.item())
            samples = model.sample(1000, sigma_1=0.001, n_timesteps=10)
            plot_samples(denorm(samples.cpu()), f"outputs/samples_{epoch}.png")


if __name__ == "__main__":

    train_loader, val_loader, denorm = make_roll_dset(int(1e4))

    model = BFN(
        dim=2,
        hidden_dims=[128, 128],
        sin_dim=16,
        time_dim=64,
        random_time_emb=False,
        dropout_p=0.0,
        device_str="cuda:0",
        dtype_str="float64",
    )
    samples = model.sample(1000, sigma_1=0.01, n_timesteps=10)
    plot_samples(denorm(samples.cpu()))

    train(model, train_loader, val_loader, denorm)
