import pytest
import torch
import torch.nn as nn

from docsaidkit.torch.nn import SeparableConvBlock


@pytest.fixture
def cnn_arch():
    return [
        {'in_channels': 3, 'out_channels': 32, 'kernel': 3},
        {'in_channels': 32, 'out_channels': 64, 'kernel': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel': 3},
    ]


@pytest.fixture
def fc_arch():
    return [
        {'in_channels': 3, 'out_channels': 32},
        {'in_channels': 32, 'out_channels': 64},
        {'in_channels': 64, 'out_channels': 128},
    ]


def test_SeparableConvBlock_forward():
    # Test input and output shapes
    in_channels = 64
    out_channels = 128
    block = SeparableConvBlock(in_channels, out_channels)
    x = torch.randn(1, in_channels, 64, 64)
    output = block(x)
    assert output.shape == (1, out_channels, 64, 64)

    # Test with different kernel size and padding
    kernel_size = (5, 3)
    padding = (1, 2)
    block = SeparableConvBlock(in_channels, out_channels, kernel=kernel_size, padding=padding)
    output = block(x)
    assert output.shape == (1, out_channels, 62, 66)

    # Test with different stride
    stride = 2
    block = SeparableConvBlock(in_channels, out_channels, stride=stride)
    output = block(x)
    assert output.shape == (1, out_channels, 32, 32)

    # Test with different output channels
    out_channels = 32
    block = SeparableConvBlock(in_channels, out_channels)
    output = block(x)
    assert output.shape == (1, out_channels, 64, 64)

    # Test without normalization and activation
    block = SeparableConvBlock(in_channels, out_channels, norm=None, act=None)
    output = block(x)
    assert output.shape == (1, out_channels, 64, 64)


def test_SeparableConvBlock_build_activation():
    # Test build_activation() function with different activation functions
    activation_fns = [
        {'name': 'ReLU'},
        {'name': 'Sigmoid'},
        {'name': 'Tanh'},
        {'name': 'LeakyReLU', 'negative_slope': 0.2}
    ]
    for act in activation_fns:
        block = SeparableConvBlock(64, 64, act=act)
        assert isinstance(block.act, nn.Module)


def test_SeparableConvBlock_build_norm():
    # Test build_norm() function with different normalization layers
    norm_layers = [
        {'name': 'BatchNorm2d', 'num_features': 64},
        {'name': 'InstanceNorm2d', 'num_features': 64},
        {'name': 'GroupNorm', 'num_groups': 8, 'num_channels': 64},
    ]
    for norm in norm_layers:
        block = SeparableConvBlock(64, 64, norm=norm)
        assert isinstance(block.norm, nn.Module)
