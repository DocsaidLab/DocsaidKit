import pytest
import torch

from docsaidkit.torch.nn import ASPPLayer, Hswish


@pytest.fixture
def input_tensor():
    return torch.randn(1, 64, 32, 32)


def test_aspp_layer(input_tensor):
    in_channels = input_tensor.size(1)
    out_channels = 128

    # Test default activation function (ReLU)
    aspp_layer = ASPPLayer(in_channels, out_channels)
    output = aspp_layer(input_tensor)
    assert output.size() == (1, out_channels, 32, 32)

    # Test with Hswish activation function
    aspp_layer = ASPPLayer(in_channels, out_channels, output_activate=Hswish())
    output = aspp_layer(input_tensor)
    assert output.size() == (1, out_channels, 32, 32)

    # Test with different dilation rates
    aspp_layer = ASPPLayer(in_channels, out_channels)
    aspp_layer.layers['DILATE1'].dilation = (2, 2)
    aspp_layer.layers['DILATE2'].dilation = (4, 4)
    aspp_layer.layers['DILATE3'].dilation = (8, 8)
    aspp_layer.layers['DILATE4'].dilation = (16, 16)
    output = aspp_layer(input_tensor)
    assert output.size() == (1, out_channels, 32, 32)
