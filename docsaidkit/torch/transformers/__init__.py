import fnmatch
from functools import partial

from .basic import ImageEncoder, ImageEncoderLayer
from .efficientformer import EfficientFormer
from .metaformer import MetaFormer, MlpBlock
from .mobilevit import MobileViT
from .poolformer import PoolFormer
from .token_mixer import (Attention, AttentionMixing, PoolMixing, RandomMixing,
                          SepConvMixing)
from .utils import calculate_patch_size, list_models_transformers
from .vit import ViT

__all__ = [
    'ViT', 'calculate_patch_size', 'list_models_transformers', 'MobileViT',
    'PoolFormer', 'MetaFormer', 'MlpBlock', 'build_transformer', 'list_transformer',
    'Attention', 'AttentionMixing', 'PoolMixing', 'RandomMixing', 'SepConvMixing',
    'ImageEncoder', 'ImageEncoderLayer', 'EfficientFormer',
]

BASE_TRANSFORMER_NAMES = {
    'vit': ViT,
    'mobilevit': MobileViT,
    'poolformer': PoolFormer,
    'metaformer': MetaFormer,
    'efficientformer': EfficientFormer,
}

VIT_NAMES = [
    'vit-base-patch16-224-in21k',
    'vit-base-patch16-224',
    'vit-base-patch16-384',
    'vit-base-patch32-224-in21k',
    'vit-base-patch32-384',
    'vit-huge-patch14-224-in21k',
    'vit-large-patch16-224-in21k',
    'vit-large-patch16-224',
    'vit-large-patch16-384',
    'vit-large-patch32-224-in21k',
    'vit-large-patch32-384',
    'vit-hybrid-base-bit-384',
]

MOBILEVIT_NAMES = [
    'mobilevit-small',
    'mobilevit-x-small',
    'mobilevit-xx-small',
    'deeplabv3-mobilevit-small',
    'deeplabv3-mobilevit-x-small',
    'deeplabv3-mobilevit-xx-small',
]

POOLFORMER_NAMES = [
    'poolformer_m36',
    'poolformer_m48',
    'poolformer_s12',
    'poolformer_s24',
    'poolformer_s36',
]

METAFORMER_NAMES = [
    'poolformer_v2_tiny',
    'poolformer_v2_small',
    'poolformer_v2_s12',
    'poolformer_v2_s24',
    'poolformer_v2_s36',
    'poolformer_v2_m36',
    'poolformer_v2_m48',
    'convformer_s18',
    'convformer_s36',
    'convformer_m36',
    'convformer_b36',
    'caformer_tiny',
    'caformer_small',
    'caformer_s18',
    'caformer_s36',
    'caformer_m36',
    'caformer_b36',
]

EFFICIENTFORMER_NAMES = [
    'efficientformer-l1-300',
    'efficientformer-l3-300',
    'efficientformer-l7-300',
]

TRANSFORMER = {
    **{name: module for name, module in BASE_TRANSFORMER_NAMES.items()},
    **{name: partial(ViT.from_pretrained, name=f'google/{name}') for name in VIT_NAMES},
    **{name: partial(MobileViT.from_pretrained, name=f'apple/{name}') for name in MOBILEVIT_NAMES},
    **{name: partial(PoolFormer.from_pretrained, name=f'sail/{name}') for name in POOLFORMER_NAMES},
    **{name: partial(MetaFormer.from_pretrained, name=name) for name in METAFORMER_NAMES},
    **{name: partial(EfficientFormer.from_pretrained, name=f'snap-research/{name}') for name in EFFICIENTFORMER_NAMES},
}


def build_transformer(name: str, **kwargs):
    if name not in TRANSFORMER:
        raise ValueError(f'Transformer={name} is not supported.')
    return TRANSFORMER[name](**kwargs)


def list_transformer(filter=''):
    model_list = list(TRANSFORMER.keys())
    if len(filter):
        return fnmatch.filter(model_list, filter)  # include these models
    else:
        return model_list
