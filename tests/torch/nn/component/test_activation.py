import pytest
import torch

from docsaidkit.torch.nn.components import (SquaredReLU, StarReLU,
                                            build_activation)

test_build_activation_data = [
    ('ReLU', torch.nn.ReLU),
    ('LeakyReLU', torch.nn.LeakyReLU),
    ('Swish', torch.nn.SiLU),
    ('StarReLU', StarReLU),
    ('SquaredReLU', SquaredReLU),
    ('FakeActivation', ValueError)
]


@pytest.mark.parametrize('name, expected_output', test_build_activation_data)
def test_build_activation(name, expected_output):
    if expected_output == ValueError:
        with pytest.raises(ValueError):
            build_activation(name)
    else:
        assert isinstance(build_activation(name), expected_output)


def test_starrelu():
    x = torch.tensor([-1, 0, 1], dtype=torch.float32)
    relu = StarReLU(scale=2.0, bias=1.0)
    # Test forward pass
    expected_output = torch.tensor([1, 1, 3], dtype=torch.float32)
    assert torch.allclose(relu(x), expected_output)

    # Test backward pass with scale and bias learnable
    optimizer = torch.optim.SGD(relu.parameters(), lr=0.01)
    loss = (relu(x).sum() - expected_output.sum()) ** 2
    loss.backward()
    optimizer.step()
    assert relu.scale.requires_grad
    assert relu.bias.requires_grad
    assert torch.allclose(relu(x), expected_output, rtol=1e-3)

    # Test backward pass with fixed scale and bias
    relu = StarReLU(scale=2.0, bias=1.0, scale_learnable=False,
                    bias_learnable=False)
    assert not relu.scale.requires_grad
    assert not relu.bias.requires_grad
    assert torch.allclose(relu(x), expected_output, rtol=1e-3)
