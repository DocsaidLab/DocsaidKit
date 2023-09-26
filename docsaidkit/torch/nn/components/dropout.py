from typing import Union

import torch.nn as nn
from torch.nn import AlphaDropout, Dropout, Dropout2d, Dropout3d

__all__ = [
    'Dropout', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'build_dropout',
]


def build_dropout(name, **options) -> Union[nn.Module, None]:
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'Dropout named {name} is not support.')
    return cls(**options)
