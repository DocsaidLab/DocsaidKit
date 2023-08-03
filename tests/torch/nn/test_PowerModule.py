import pytest
import torch

from docsaidkit.torch.nn.utils import PowerModule, initialize_weights


class SimpleModel(PowerModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)
        self.layer2 = torch.nn.Linear(5, 2)


@pytest.fixture
def model():
    return SimpleModel()


def test_initialize_weights(model):
    initialize_weights(model)
    for param in model.parameters():
        assert not torch.isnan(param).any()


def test_freeze(model):
    model.freeze(verbose=True)
    for param in model.parameters():
        assert not param.requires_grad


def test_melt(model):
    model.freeze(verbose=True)
    model.melt(verbose=True)
    for param in model.parameters():
        assert param.requires_grad


def test_initialize_weights_(model):
    model.initialize_weights_()
    for param in model.parameters():
        assert not torch.isnan(param).any()


def test_freeze_layer(model):
    model.freeze('layer1', verbose=True)
    assert not model.layer1.weight.requires_grad


def test_melt_layer(model):
    model.freeze('layer1', verbose=True)
    model.melt('layer1', verbose=True)
    assert model.layer1.weight.requires_grad
