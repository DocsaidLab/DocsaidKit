import pytest
import torch
import torch.nn as nn

from docsaidkit.torch.nn import WeightedSum


def test_weighted_sum_init():
    input_size = 3
    ws = WeightedSum(input_size)
    assert ws.input_size == input_size
    assert isinstance(ws.weights, nn.Parameter)
    assert ws.weights.shape == (input_size,)
    assert ws.relu.__class__.__name__ == 'Identity'
    assert ws.epsilon == 1e-4


def test_weighted_sum_forward():
    input_size = 3
    ws = WeightedSum(input_size)

    # Test valid input
    x = [torch.randn(1, 5) for _ in range(input_size)]
    y = ws(*x)
    assert y.shape == (1, 5)
    assert torch.allclose(y, torch.mean(torch.cat(x), dim=0, keepdim=True), atol=1e-3)

    # Test invalid input size
    with pytest.raises(ValueError):
        ws(*x[:-1])

    # Test activation function
    max_v = 10
    min_v = -10
    ws = WeightedSum(input_size, act=nn.ReLU(False))
    x = [(max_v - min_v) * torch.rand(1, 5) + min_v for _ in range(input_size)]
    y = ws(*x)
    assert torch.allclose(y, torch.mean(torch.cat(x), dim=0, keepdim=True).relu(), atol=1e-3)

    # Test custom activation function
    class custom_act(nn.Module):
        def forward(self, x):
            return x + 1

    ws = WeightedSum(input_size, act=custom_act())
    x = [torch.randn(1, 5) for _ in range(input_size)]
    y = ws(*x)
    assert torch.allclose(y, custom_act()(torch.mean(torch.cat(x), dim=0, keepdim=True)), atol=1e-3)
