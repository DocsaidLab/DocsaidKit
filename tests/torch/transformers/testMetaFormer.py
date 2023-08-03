import pytest
import torch
from torch import nn

from docsaidkit.torch.transformers.metaformer import (MetaFormer,
                                                      MetaFormerBlock)


def test_init():
    model = MetaFormer()
    assert isinstance(model, nn.Module)
    assert isinstance(model, MetaFormer)


def test_forward():
    model = MetaFormer()
    input_tensor = torch.rand(3, 3, 224, 224)
    all_hidden_state = model(input_tensor)
    assert all_hidden_state[0].shape == torch.Size([3, 64, 56, 56])
    assert all_hidden_state[1].shape == torch.Size([3, 128, 28, 28])
    assert all_hidden_state[2].shape == torch.Size([3, 320, 14, 14])
    assert all_hidden_state[3].shape == torch.Size([3, 512, 7, 7])


def test_token_mixer():
    model = MetaFormer(token_mixers=[
        {'name': 'AttentionMixing', 'in_features': 64},
        {'name': 'PoolMixing', 'pool_size': 5},
        {'name': 'RandomMixing', 'num_tokens': 196},
        {'name': 'SepConvMixing', 'in_features': 512, 'expand_ratio': 4}
    ])
    input_tensor = torch.rand(3, 3, 224, 224)
    all_hidden_state = model(input_tensor)
    assert all_hidden_state[0].shape == torch.Size([3, 64, 56, 56])
    assert all_hidden_state[1].shape == torch.Size([3, 128, 28, 28])
    assert all_hidden_state[2].shape == torch.Size([3, 320, 14, 14])
    assert all_hidden_state[3].shape == torch.Size([3, 512, 7, 7])



@pytest.fixture
def input_tensor():
    return torch.randn(2, 3, 16, 16)

@pytest.fixture
def metaformer_block():
    return MetaFormerBlock(3)

def test_metaformer_block_forward(metaformer_block, input_tensor):
    output = metaformer_block(input_tensor)
    assert output.shape == input_tensor.shape
