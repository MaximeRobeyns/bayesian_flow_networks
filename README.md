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

> Note: work in progress. More examples to come.
