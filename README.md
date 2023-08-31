# Bayesian Flow Networks in PyTorch

A PyTorch implementation of [Bayesian Flow Networks (Graves et al., 2023)](https://arxiv.org/abs/2308.07037).

See my explanatory blog post [here](https://maximerobeyns.com/bayesian_flow_networks).

## Installation

```bash
git clone https://github.com/MaximeRobeyns/bayesian_flow_networks
cd bayesian_flow_networks
pip install -e .
```

## Examples

### Continuous Data (swiss roll)

Both the infinite and discrete time loss functions are implemented.

Here is a minimal example for the 2D swiss roll dataset (see
`examples/swiss_roll_bfn.py` for the full code). Here are some model samples
throughout training:

![Swiss roll samples throughout training](./examples/swiss_roll.png)

```python
# Imports
import torch
from torch_bfn import ContinuousBFN, LinearNetwork
from torch_bfn.utils import EMA

# Setup a suitable network
net = LinearNetwork(dim=2, hidden_dims=[512, 512], sin_dim=16, time_dim=64)

# Setup the BFN
model = ContinuousBFN(dim=2, net=net)

# Setup training
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ema = EMA(0.9)
ema.register(model)

# Load data (see examples/swiss_roll_bfn)
train_loader = ...

# Train the model
for epoch in range(100):
    for batch in train_loader:
        X = batch[0].to(device, dtype)
        # For continuous loss:
        loss = model.loss(X, sigma_1=0.01).mean()
        # For discrete-time loss:
        # loss = model.discrete_loss(X, sigma_1=0.01, n=30).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

# Sample from the model
samples = model.sample(1000, sigma_1=0.01, n_timesteps=10)
```

## Classifier-Free Guidance with Continuous Data (MNIST)

In `examples/MNIST_continuous_bfn.py`, we show an example training a UNet on
MNIST with classifier-free guidance.

![MNIST samples with classifier-free guidance](./mnist_cont_classifier_free_guidance.png)

```python

# Get data loader (see examples/MNIST_continuous_bfn.py) for full code
train_loader = get_mnist()

# Create the UNet for MNIST
net = Unet(
    dim=128,
    channels=1,
    dim_mults=[1, 2, 2],
    num_classes=10,
    cond_drop_prob=0.5,
    flash_attn=True,
)

# Create the BFN
model = ContinuousBFN(dim=(1, 28, 28), net=net)

# Setup training
ema = EMA(0.99)
opt = t.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.98)
)
ema.register(model)

# Run training loop
for epoch in range(epochs):
    for batch in train_loader:
        X, y = batch
        # Continuous-time loss
        loss = model.loss(X, y, sigma_1=0.01).mean()
        # Discrete-time loss
        # loss = model.discrete_loss(*batch, sigma_1=0.01, n=30).mean()
        opt.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
```

> Note: work in progress. More examples to come.
