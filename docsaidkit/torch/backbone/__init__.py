import fnmatch
from functools import partial

from timm import create_model, list_models

from .gpunet import GPUNet

__all__ = [
    'BACKBONE', 'build_backbone', 'list_backbones',
]

GPUNET_NAMES = [
    'gpunet_0',
    'gpunet_1',
    'gpunet_2',
    'gpunet_p0',
    'gpunet_p1',
    'gpunet_d1',
    'gpunet_d2',
]


BACKBONE = {
    **{k: partial(create_model, model_name=k) for k in list_models()},
    **{name: partial(GPUNet.build_gpunet, name=name) for name in GPUNET_NAMES},
}


def build_backbone(name: str, **kwargs):
    if name not in BACKBONE:
        raise ValueError(f'Backbone={name} is not supported.')
    return BACKBONE[name](**kwargs)


def list_backbones(filter=''):
    model_list = list(BACKBONE.keys())
    if len(filter):
        return fnmatch.filter(model_list, filter)  # include these models
    else:
        return model_list
