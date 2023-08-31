import torch as t

from torch_bfn.networks import Unet


def test_unet():
    model = Unet(128, channels=1)
    input_tensor = t.randn(8, 1, 28, 28)
    output = model(input_tensor, t.tensor([0.5]))
    assert output.shape == (8, 1, 28, 28)


def test_class_conditional_unet():
    model = Unet(128, channels=1, num_classes=10, cond_drop_prob=0.2)
    input_tensor = t.randn(8, 1, 28, 28)
    class_tensor = t.randint(0, 10, (8, 1))
    output = model(input_tensor, t.tensor([0.5]), class_tensor)
    class_tensor = t.randint(0, 10, (8,))
    output = model(input_tensor, t.tensor([0.5]), class_tensor)
    assert output.shape == (8, 1, 28, 28)


def test_cond_unet():
    model = Unet(128, channels=1, num_classes=10)
    input_tensor = t.randn(8, 1, 28, 28)
    input_classes = t.randint(0, 10, (8,))
    output = model(input_tensor, t.tensor([0.5]), input_classes)
    output = model(input_tensor, t.tensor([0.5]).expand(8), input_classes)
    assert output.shape == (8, 1, 28, 28)
