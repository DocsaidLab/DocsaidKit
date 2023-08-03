import pytest
import torch
import torch.nn as nn

from docsaidkit.torch.nn import CNN2Dcell


@pytest.fixture
def input_tensor():
    return torch.randn((2, 3, 32, 32))


@pytest.fixture
def output_shape():
    return (2, 16, 32, 32)


def test_cnn2dcell_forward(input_tensor, output_shape):
    model = CNN2Dcell(in_channels=3, out_channels=16)
    output = model(input_tensor)
    assert output.shape == output_shape


def test_cnn2dcell_with_activation(input_tensor, output_shape):
    model = CNN2Dcell(in_channels=3, out_channels=16, act={'name': 'ReLU', 'inplace': True})
    output = model(input_tensor)
    assert output.shape == output_shape
    assert torch.all(output >= 0)


def test_cnn2dcell_with_batch_norm(input_tensor, output_shape):
    model = CNN2Dcell(in_channels=3, out_channels=16, norm={'name': 'BatchNorm2d', 'num_features': 16})
    output = model(input_tensor)
    assert output.shape == output_shape
    assert torch.allclose(output.mean(dim=(0, 2, 3)), torch.zeros(16), rtol=1e-3, atol=1e-5)
    assert torch.allclose(output.var(dim=(0, 2, 3)), torch.ones(16), rtol=1e-3, atol=1e-5)


def test_cnn2dcell_with_dropout(input_tensor, output_shape):
    model = CNN2Dcell(in_channels=3, out_channels=16, dropout={'name': 'Dropout2d', 'p': 0.5})
    output = model(input_tensor)
    assert output.shape == output_shape


def test_cnn2dcell_with_pooling(input_tensor):
    model = CNN2Dcell(in_channels=3, out_channels=16, pool=nn.AdaptiveAvgPool2d(1))
    output = model(input_tensor)
    assert output.shape == (2, 16, 1, 1)


def test_cnn2dcell_init_type(input_tensor):
    model = CNN2Dcell(in_channels=3, out_channels=16, init_type='uniform')
    output1 = model(input_tensor)
    model = CNN2Dcell(in_channels=3, out_channels=16, init_type='normal')
    output2 = model(input_tensor)
    assert not torch.allclose(output1, output2, rtol=1e-3, atol=1e-5)


def test_cnn2dcell_all_together(input_tensor):
    model = CNN2Dcell(in_channels=3, out_channels=16,
                      kernel=5, stride=2, padding=2, dilation=2, groups=1,
                      bias=True, padding_mode='reflect',
                      norm={'name': 'BatchNorm2d', 'num_features': 16, 'momentum': 0.5},
                      dropout={'name': 'Dropout2d', 'p': 0.5},
                      act={'name': 'LeakyReLU', 'negative_slope': 0.1, 'inplace': True},
                      pool=nn.AdaptiveAvgPool2d(1),
                      init_type='uniform')
    output = model(input_tensor)
    assert output.shape == (2, 16, 1, 1)
