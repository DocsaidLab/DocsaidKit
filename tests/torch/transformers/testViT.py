import torch
from torch import nn
from transformers import ViTConfig, ViTModel

from docsaidkit.torch import ViT


def test_init():
    model = ViT()
    assert isinstance(model, nn.Module)
    assert isinstance(model.config, ViTConfig)
    assert isinstance(model.model, ViTModel)


def test_forward():
    model = ViT()
    input_tensor = torch.rand(1, 3, 224, 224)
    cls_token, hidden_state = model(input_tensor)
    assert cls_token.shape == torch.Size([1, 768])
    assert hidden_state.shape == torch.Size([1, 196, 768])


def test_list_pretrained_models():
    models = ViT.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_from_pretrained():
    model = ViT.from_pretrained('google/vit-base-patch16-224')
    input_tensor = torch.rand(1, 3, 224, 224)
    cls_token, hidden_state = model(input_tensor)
    assert cls_token.shape == torch.Size([1, 768])
    assert hidden_state.shape == torch.Size([1, 196, 768])
