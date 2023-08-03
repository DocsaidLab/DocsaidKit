import pytest
import torch

from docsaidkit.torch.nn import VAE


@pytest.fixture
def input_tensor():
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def vae():
    return VAE(3 * 32 * 32, 100)


def test_vae_forward_shape(input_tensor, vae):
    feat, kld_loss = vae(input_tensor.view(input_tensor.size(0), -1))
    assert feat.shape == (input_tensor.size(0), 100)
    assert kld_loss.shape == ()


def test_vae_forward_kld_loss(input_tensor, vae):
    feat, kld_loss = vae(input_tensor.view(input_tensor.size(0), -1))
    assert kld_loss >= 0


def test_vae_do_pooling():
    model = VAE(in_channels=10, out_channels=5, do_pooling=True)
    x = torch.randn(2, 10, 64, 64)
    feat, kld_loss = model(x)
    assert feat.shape == (2, 5)
    assert kld_loss.shape == ()
