from torch.nn import *

from .aspp import *
from .block import *
from .cnn import *
from .components import *
from .dwcnn import *
from .grl import *
from .mbcnn import *
from .positional_encoding import *
from .selayer import *
from .utils import *
from .vae import *


def build_nn_cls(name):
    cls_ = globals().get(name, None)
    if cls_ is None:
        raise ImportError(f'name {name} is not in nn.')
    return cls_


def build_nn(name, **kwargs):
    return build_nn_cls(name)(**kwargs)
