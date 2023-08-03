import pytest
import torch

from docsaidkit.torch import build_pool


@pytest.fixture
def input_tensor():
    return torch.randn(2, 3, 16, 16)


pool_layers = [
    ('AdaptiveAvgPool2d', {'output_size': (1, 1)}, (2, 3, 1, 1)),
    ('AdaptiveMaxPool2d', {'output_size': (1, 1)}, (2, 3, 1, 1)),
    ('AvgPool2d', {'kernel_size': 3, 'stride': 1, 'padding': 1}, (2, 3, 16, 16)),
    ('MaxPool2d', {'kernel_size': 3, 'stride': 1, 'padding': 1}, (2, 3, 16, 16)),
    ('GAP', {}, (2, 3)),
    ('GMP', {}, (2, 3)),
]


@pytest.mark.parametrize('name, kwargs, expected_shape', pool_layers)
def test_pool_layer(name, kwargs, expected_shape, input_tensor):
    # Build the pool layer
    layer = build_pool(name, **kwargs)

    # Check the output shape
    output = layer(input_tensor)
    assert output.shape == expected_shape
