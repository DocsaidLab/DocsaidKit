import torch
import torch.nn as nn

from docsaidkit.torch.nn import MBCNNcell


def test_mbcnncell_identity():
    # Test identity block
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell(16, 16, kernel=3, stride=1)
    out = block(x)
    assert out.shape == x.shape


def test_mbcnncell_expdim():
    # Test expansion block
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell(16, 32, kernel=3, stride=1)
    out = block(x)
    assert out.shape == (1, 32, 32, 32)


def test_mbcnncell_norm():
    # Test block with normalization layer
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell(16, 16, kernel=3, stride=1, norm=nn.BatchNorm2d(16))
    out = block(x)
    assert out.shape == x.shape


def test_mbcnncell_se():
    # Test block with Squeeze-and-Excitation layer
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell(16, 16, kernel=3, stride=1, use_se=True)
    out = block(x)
    assert out.shape == x.shape


def test_mbcnncell_build_mbv1block():
    # Test building of MobileNetV1-style block
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell.build_mbv1block(16, 32)
    out = block(x)
    assert out.shape == (1, 32, 32, 32)


def test_mbcnncell_build_mbv2block():
    # Test building of MobileNetV2-style block
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell.build_mbv2block(16, 32)
    out = block(x)
    assert out.shape == (1, 32, 32, 32)


def test_mbcnncell_build_mbv3block():
    # Test building of MobileNetV3-style block
    x = torch.randn(1, 16, 32, 32)
    block = MBCNNcell.build_mbv3block(16, 32)
    out = block(x)
    assert out.shape == (1, 32, 32, 32)
