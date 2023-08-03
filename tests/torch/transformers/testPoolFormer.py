import torch
from torch import nn
from transformers import PoolFormerConfig, PoolFormerModel

from docsaidkit.torch import PoolFormer


def test_init():
    model = PoolFormer()
    assert isinstance(model, nn.Module)
    assert isinstance(model.config, PoolFormerConfig)
    assert isinstance(model.model, PoolFormerModel)


def test_forward():
    model = PoolFormer()
    input_tensor = torch.rand(1, 3, 224, 224)
    all_hidden_state = model(input_tensor)
    assert all_hidden_state[0].shape == torch.Size([1, 64, 56, 56])
    assert all_hidden_state[1].shape == torch.Size([1, 128, 28, 28])
    assert all_hidden_state[2].shape == torch.Size([1, 320, 14, 14])
    assert all_hidden_state[3].shape == torch.Size([1, 512, 7, 7])


def test_list_pretrained_models():
    models = PoolFormer.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_from_pretrained():
    model = PoolFormer.from_pretrained('sail/poolformer_s12')
    input_tensor = torch.rand(1, 3, 224, 224)
    all_hidden_state = model(input_tensor)
    assert all_hidden_state[0].shape == torch.Size([1, 64, 56, 56])
    assert all_hidden_state[1].shape == torch.Size([1, 128, 28, 28])
    assert all_hidden_state[2].shape == torch.Size([1, 320, 14, 14])
    assert all_hidden_state[3].shape == torch.Size([1, 512, 7, 7])
