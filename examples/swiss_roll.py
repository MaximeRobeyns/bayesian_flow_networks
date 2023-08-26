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
"""Swiss roll example"""

import torch as t
import matplotlib.pyplot as plt

from typing import Callable, Tuple
from torchtyping import TensorType as Tensor
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch_bfn import DDPM
from torch_bfn.utils import EMA, norm_denorm, str_to_torch_dtype


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


def plot_fwd_diffusion(data: Tensor["B", "D"], model: DDPM):
    _, axs = plt.subplots(1, 10, figsize=(30, 3))
    for i in range(10):
        q_i = model.q_sample(data, t.tensor([i * 10]))
        axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
        axs[i].set_axis_off()
        axs[i].set_title("$q(\\mathbf{x}_{" + str(i * 10) + "})$")
    plt.savefig("outputs/fwd.png")
    plt.close()


def train(
    model: DDPM,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
            loss = model.noise_estimation_loss(X)
            opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        if epoch % 200 == 0:
            assert loss is not None
            print(loss.item())
            x_seq = model.p_sample_loop(val_loader.dataset[:][0].shape)
            fig, axs = plt.subplots(1, 10, figsize=(30, 3))
            for i in range(1, 11):
                cur_x = x_seq[i * 10].detach()
                axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], s=10)
                axs[i - 1].set_title("$q(\mathbf{x}_{" + str(i * 100) + "})$")
            plt.savefig(f"outputs/train_{epoch}.png")
            plt.close()


if __name__ == "__main__":

    train_loader, val_loader, denorm = make_roll_dset(int(1e4))

    model = DDPM()
    print(model)

    # Plot forward model for sanity
    plot_fwd_diffusion(train_loader.dataset[:][0], model)

    train(model, train_loader, val_loader)
