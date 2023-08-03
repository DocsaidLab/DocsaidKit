import torch

from docsaidkit.torch.nn import SELayer


def test_selayer_output_shape():
    # Test that the output shape of the SELayer is correct.
    in_channels = 16
    reduction = 4
    batch_size = 8
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, height, width)
    se_layer = SELayer(in_channels, reduction)
    y = se_layer(x)

    assert y.shape == x.shape


def test_selayer_activation():
    # Test that the output of the SELayer is activated correctly.
    in_channels = 16
    reduction = 4
    batch_size = 8
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, height, width)
    se_layer = SELayer(in_channels, reduction)
    y = se_layer(x)

    assert torch.all(y / x >= 0)
    assert torch.all(y / x <= 1)


def test_selayer_reduction():
    # Test that the SELayer reduces the number of channels as expected.
    in_channels = 16
    reduction = 4
    batch_size = 8
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, height, width)
    se_layer = SELayer(in_channels, reduction)
    y = se_layer(x)

    expected_channels = in_channels // reduction

    assert se_layer.fc1.layer.cnn.out_channels == expected_channels
    assert se_layer.fc2.layer.cnn.out_channels == in_channels
