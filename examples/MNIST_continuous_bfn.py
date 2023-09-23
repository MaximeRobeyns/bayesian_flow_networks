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
"""Continuous (cast to float) MNIST example with BFN"""

import os
import torch as t
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch_bfn import Unet, ContinuousBFN
from torch_bfn.utils import str_to_torch_dtype, EMA


def get_mnist() -> DataLoader:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs = cuda_kwargs | {"batch_size": 128}

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ]
    )

    dset_path = "/tmp/MNIST"
    train_set = datasets.MNIST(
        dset_path, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, **train_kwargs)
    return train_loader


def dcn(x: t.Tensor) -> np.ndarray:
    return x.squeeze(0).detach().cpu().numpy()


def plot_samples(samples: t.Tensor, fpath: str = "outputs/mnist_samples.png"):
    n = len(samples)
    fig, ax = plt.subplots(1, n, figsize=(n * 4, 4), dpi=150)
    for i in range(n):
        ax[i].imshow(dcn(samples[i]))
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    plt.savefig(fpath)
    # plt.show()
    plt.close()


def train(
    model: ContinuousBFN,
    train_loader: DataLoader,
    epochs: int = 200,
    device_str: str = "cuda:0",
    dtype_str: str = "float32",
):
    device = t.device(device_str)
    dtype = str_to_torch_dtype(dtype_str)
    ema = EMA(0.99)

    model.to(device, dtype)
    opt = t.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.98)
    )
    ema.register(model)

    for epoch in range(epochs):
        loss = None
        for batch in train_loader:
            X, y = batch
            X, y = X.to(device, dtype), y.to(device)
            loss = model.loss(X, y, sigma_1=1e-3).mean()
            # loss = model.discrete_loss(X, y, sigma_1=0.01, n=30).mean()
            opt.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        if epoch % 1 == 0:
            assert loss is not None
            print(loss.item())
            sample_classes = t.arange(10, device=device)
            samples = model.sample(
                1,
                sigma_1=1e-5,
                n_timesteps=20,
                cond=sample_classes,
                cond_scale=7.0,
            ).squeeze(1)
            plot_samples(samples, f"outputs/mnist_samples_{epoch:03d}.png")


if __name__ == "__main__":

    train_loader = get_mnist()
    device = "cuda:0"
    dtype = "float32"

    net = Unet(
        dim=512,
        channels=1,
        dim_mults=[1, 2, 2],
        num_classes=10,
        cond_drop_prob=0.5,
        flash_attn=True,
    )

    model = ContinuousBFN(
        dim=(1, 28, 28),
        net=net,
        device_str=device,
        dtype_str=dtype,
    )

    sample_classes = t.arange(10, device=t.device(device))
    samples = model.sample(
        1, sigma_1=1e-5, n_timesteps=20, cond=sample_classes, cond_scale=7.0
    ).squeeze(1)
    plot_samples(samples, "outputs/initial_mnist_samples.png")

    train(
        model,
        train_loader,
        device_str=device,
        dtype_str=dtype,
    )

    samples = model.sample(
        1, sigma_1=1e-5, n_timesteps=20, cond=sample_classes, cond_scale=7.0
    ).squeeze(1)
    plot_samples(samples, "outputs/final_mnist_samples.png")
