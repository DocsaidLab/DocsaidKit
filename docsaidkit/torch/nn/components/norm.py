from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import (BatchNorm1d, BatchNorm2d, BatchNorm3d,
                                        SyncBatchNorm)
from torch.nn.modules.instancenorm import (InstanceNorm1d, InstanceNorm2d,
                                           InstanceNorm3d)
from torch.nn.modules.normalization import (CrossMapLRN2d, GroupNorm,
                                            LayerNorm, LocalResponseNorm)

__all__ = [
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'SyncBatchNorm', 'InstanceNorm1d',
    'InstanceNorm2d', 'InstanceNorm3d', 'CrossMapLRN2d', 'GroupNorm', 'LayerNorm',
    'LocalResponseNorm', 'build_norm', 'LayerNorm2d',
]


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        r"""
        LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).

        Args:
            num_channels (int):
                Number of channels in the input tensor.
            eps (float, optional):
                A value added to the denominator for numerical stability.
                Default: 1e-5
            affine (bool. optional):
                A boolean value that when set to `True`, this module has learnable
                per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default to True.
        """
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape,
                         self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def build_norm(name: str, **options) -> Union[nn.Module, None]:
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(
            f'Normalization named {name} is not supported. Available options: {__all__}')
    return cls(**options)
