import pytest
import torch

from docsaidkit.torch import build_transformer, list_transformer
from docsaidkit.torch.transformers import (BASE_TRANSFORMER_NAMES, TRANSFORMER,
                                           EfficientFormer, MetaFormer,
                                           MobileViT, PoolFormer, ViT)


def test_list_transformer():
    models = list_transformer()
    assert len(models) == len(TRANSFORMER)


@pytest.mark.parametrize("model_name", BASE_TRANSFORMER_NAMES.keys())
def test_build_transformer(model_name):
    model = build_transformer(model_name)
    assert isinstance(model, (ViT, MobileViT, PoolFormer, MetaFormer, EfficientFormer))
