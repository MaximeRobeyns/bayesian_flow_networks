# Bayesian Flow Networks in PyTorch

A PyTorch implementation of Bayesian Flow Networks (Graves et al., 2023).

See associated blog post [here](https://maximerobeyns.com/bayesian_flow_networks).

## Installation

```bash
git clone https://github.com/MaximeRobeyns/bayesian_flow_networks
cd bayesian_flow_networks
pip install -e .
```

## Quickstart

For continuous data, with continuous-time loss

```python
# Setup the BFN
import torch
from torch_bfn import BFN
from torch_bfn.utils import EMA

model = BFN(dim=2, hidden_dims=[128, 128])

# Train the model
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ema = EMA(0.9)
ema.register(model)

for epoch in range(epochs):
    for batch in train_loader:
        X = batch[0].to(device, dtype)
        loss = model.loss(X, sigma_1=0.01).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

# Sample from the model
samples = model.sample(1000, sigma_1=0.01, n_timesteps=10)
```

> Note: work in progress. More examples to come.
