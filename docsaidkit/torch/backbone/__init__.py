import fnmatch
from functools import partial

from timm import create_model, list_models

__all__ = [
    'BACKBONE', 'build_backbone', 'list_backbones',
]

BACKBONE = {
    **{k: partial(create_model, model_name=k) for k in list_models()},
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
