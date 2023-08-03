import pytest
import torch
from torch.nn.modules.batchnorm import (BatchNorm1d, BatchNorm2d, BatchNorm3d,
                                        SyncBatchNorm)
from torch.nn.modules.instancenorm import (InstanceNorm1d, InstanceNorm2d,
                                           InstanceNorm3d)
from torch.nn.modules.normalization import (CrossMapLRN2d, GroupNorm,
                                            LayerNorm, LocalResponseNorm)

from docsaidkit.torch import LayerNorm2d, build_norm

NORM_CLASSES = {
    'BatchNorm1d': BatchNorm1d,
    'BatchNorm2d': BatchNorm2d,
    'BatchNorm3d': BatchNorm3d,
    'SyncBatchNorm': SyncBatchNorm,
    'InstanceNorm1d': InstanceNorm1d,
    'InstanceNorm2d': InstanceNorm2d,
    'InstanceNorm3d': InstanceNorm3d,
    'CrossMapLRN2d': CrossMapLRN2d,
    'GroupNorm': GroupNorm,
    'LayerNorm': LayerNorm,
    'LayerNorm2d': LayerNorm2d,
    'LocalResponseNorm': LocalResponseNorm,
}


@pytest.mark.parametrize('name', NORM_CLASSES.keys())
def test_build_norm(name: str) -> None:
    options = {}
    cls = NORM_CLASSES[name]
    if name.startswith('BatchNorm'):
        options['num_features'] = 8
    elif name.startswith('InstanceNorm'):
        options['num_features'] = 8
    elif name.startswith('SyncBatchNorm'):
        options['num_features'] = 8
    elif name.startswith('GroupNorm'):
        options['num_groups'] = 4
        options['num_channels'] = 8
    elif name.startswith('LayerNorm'):
        if name == 'LayerNorm2d':
            options['num_channels'] = 8
        else:
            options['normalized_shape'] = [8]
    elif name.startswith('LocalResponseNorm'):
        options['size'] = 3
    elif name.startswith('CrossMapLRN2d'):
        options['size'] = 3
    norm = build_norm(name, **options)
    assert isinstance(norm, cls)


def test_layer_norm_2d():
    # Create a tensor of size (N, C, H, W)
    x = torch.randn(2, 3, 4, 4)
    # Initialize LayerNorm2d
    ln = LayerNorm2d(num_channels=3)
    # Forward pass
    y = ln(x)
    # Check output shape
    assert y.shape == (2, 3, 4, 4)
