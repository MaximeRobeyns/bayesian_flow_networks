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
"""Modelling a 'two moons' dataset with the simple linear network provided with
`torch_bfn` and using classifier-free guidance.
"""

import os
import torch as t
import matplotlib.pyplot as plt

from typing import Callable, Tuple
from torchtyping import TensorType as Tensor
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch_bfn import ContinuousBFN, LinearNetwork
from torch_bfn.utils import EMA, norm_denorm, str_to_torch_dtype


def make_moon_dset(
    n: int, bs: int = 128, noise: float = 0.1, dtype: t.dtype = t.float32
) -> Tuple[
    DataLoader, DataLoader, Callable[[Tensor["B", "D"]], Tensor["B", "D"]]
]:
    # Create a normalised 'two moons' dataset:
    X_np, y_np = make_moons(n_samples=n, noise=noise)
    X, y = t.tensor(X_np, dtype=dtype), t.tensor(y_np, dtype=t.int)

    X, denorm = norm_denorm(X)

    dset = TensorDataset(X, y)
    train_size = len(dset) - bs  # - num_val
    val_size = len(dset) - train_size

    train_dset, val_dset = random_split(dset, [train_size, val_size])
    train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=bs, shuffle=False)
    return train_loader, val_loader, denorm


def plot_samples(
    denormed_xs: Tensor["B", "D"],
    ys: Tensor["B"],
    fpath: str = "outputs/moon_samples.png",
):
    xs, ys = denormed_xs.numpy(), ys.numpy()
    plt.figure(figsize=(6, 4))
    plt.scatter(*xs.T, c=ys, cmap=plt.cm.coolwarm, edgecolor="k")
    plt.title("Two Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    plt.savefig(fpath)
    plt.close()


def train(
    model: ContinuousBFN,
    train_loader: DataLoader,
    denorm: Callable[[Tensor["B", "D"]], Tensor["B", "D"]],
    epochs: int = 100,
    device_str: str = "cuda:0",
    dtype_str: str = "float32",
):
    device = t.device(device_str)
    dtype = str_to_torch_dtype(dtype_str)
    ema = EMA(0.9)

    model.to(device, dtype)
    opt = t.optim.AdamW(model.parameters(), lr=1e-3)
    ema.register(model)

    for epoch in range(epochs):
        loss = None
        for batch in train_loader:
            X, y = batch
            X, y = X.to(device, dtype), y.to(device)
            loss = model.loss(X, y, sigma_1=0.01).mean()
            # loss = model.discrete_loss(X, y, sigma_1=0.01, n=30).mean()
            opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        if epoch % 20 == 0:
            assert loss is not None
            print(loss.item())
            class_samples = t.randint(0, 2, (1000,)).to(device)
            samples = model.sample(1, cond=class_samples).squeeze(1)
            plot_samples(
                denorm(samples.cpu()).detach(),
                class_samples.cpu(),
                f"outputs/moons_{epoch:03d}",
            )


if __name__ == "__main__":
    train_loader, val_loader, denorm = make_moon_dset(int(1e4))

    # Plot the first 500 points
    plot_samples(*train_loader.dataset[:500])

    device = "cuda:0"
    dtype = "float32"

    net = LinearNetwork(
        dim=2,
        hidden_dims=[512, 512],
        cond_dim=2,
        sin_dim=16,
        time_dim=64,
        random_time_emb=False,
        dropout_p=0.0,
    ).to(device)

    model = ContinuousBFN(dim=2, net=net, device_str=device, dtype_str=dtype)

    class_samples = t.randint(0, 2, (1000,)).to(device)
    samples = model.sample(1, cond=class_samples, cond_scale=1.2).squeeze(1)
    plot_samples(
        denorm(samples.cpu()).detach(),
        class_samples.cpu(),
        "outputs/moons_prior_samples.png",
    )

    train(model, train_loader, denorm, 100, device, dtype)

    samples = model.sample(1, cond=class_samples, cond_scale=1.2).squeeze(1)
    plot_samples(
        denorm(samples.cpu()).detach(),
        class_samples.cpu(),
        "outputs/moons_final_samples.png",
    )
