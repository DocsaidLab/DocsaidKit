import fnmatch

from .bifpn import BiFPN, BiFPNs
from .fpn import FPN, FPNs

NECK = {
    'fpn': FPN,
    'fpns': FPNs,
    'bifpn': BiFPN,
    'bifpns': BiFPNs,
}

__all__ = [
    'NECK', 'BiFPN', 'BiFPNs', 'FPN', 'FPNs', 'build_neck', 'list_necks',
]

def build_neck(name: str, **kwargs):
    if name in NECK:
        neck = NECK[name](**kwargs)
    else:
        raise ValueError(f'Neck={name} is not support.')

    return neck


def list_necks(filter=''):
    model_list = list(NECK.keys())
    if len(filter):
        return fnmatch.filter(model_list, filter)  # include these models
    else:
        return model_list
