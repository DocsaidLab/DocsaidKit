import torch
from torch import nn
from transformers import MobileViTConfig, MobileViTModel

from docsaidkit.torch import MobileViT


def test_init():
    model = MobileViT()
    assert isinstance(model, nn.Module)
    assert isinstance(model.config, MobileViTConfig)
    assert isinstance(model.model, MobileViTModel)


def test_forward():
    model = MobileViT()
    input_tensor = torch.rand(1, 3, 224, 224)
    all_hidden_state = model(input_tensor)
    assert all_hidden_state[0].shape == torch.Size([1, 32, 112, 112])
    assert all_hidden_state[1].shape == torch.Size([1, 64, 56, 56])
    assert all_hidden_state[2].shape == torch.Size([1, 96, 28, 28])
    assert all_hidden_state[3].shape == torch.Size([1, 128, 14, 14])
    assert all_hidden_state[4].shape == torch.Size([1, 160, 7, 7])


def test_list_pretrained_models():
    models = MobileViT.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_from_pretrained():
    model = MobileViT.from_pretrained('apple/mobilevit-small')
    input_tensor = torch.rand(1, 3, 224, 224)
    all_hidden_state = model(input_tensor)
    assert all_hidden_state[0].shape == torch.Size([1, 32, 112, 112])
    assert all_hidden_state[1].shape == torch.Size([1, 64, 56, 56])
    assert all_hidden_state[2].shape == torch.Size([1, 96, 28, 28])
    assert all_hidden_state[3].shape == torch.Size([1, 128, 14, 14])
    assert all_hidden_state[4].shape == torch.Size([1, 160, 7, 7])
