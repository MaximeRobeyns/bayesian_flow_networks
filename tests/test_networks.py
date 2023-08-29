import torch as t

from torch_bfn.networks import Unet


def test_unet():
    model = Unet(128, channels=1)
    input_tensor = t.randn(8, 1, 28, 28)
    output = model(input_tensor, t.tensor([0.5]))
    assert output.shape == (8, 1, 28, 28)
