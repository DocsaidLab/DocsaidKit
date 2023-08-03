from typing import Union

import torch
from torch import nn
from torch.nn.modules.pooling import (AdaptiveAvgPool1d, AdaptiveAvgPool2d,
                                      AdaptiveAvgPool3d, AdaptiveMaxPool1d,
                                      AdaptiveMaxPool2d, AdaptiveMaxPool3d,
                                      AvgPool1d, AvgPool2d, AvgPool3d,
                                      MaxPool1d, MaxPool2d, MaxPool3d)

__all__ = [
    'build_pool', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'MaxPool1d',
    'MaxPool2d', 'MaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d', 'GAP', 'GMP',
]


class GAP(nn.Module):
    """Global Average Pooling layer."""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global average pooling on the input tensor."""
        return self.pool(x)


class GMP(nn.Module):
    """Global Max Pooling layer."""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global max pooling on the input tensor."""
        return self.pool(x)


def build_pool(name: str, **options) -> Union[nn.Module, None]:
    """Build a pooling layer given the name and options."""
    cls = globals().get(name, None)
    if cls is None:
        raise KeyError(f'Unsupported pooling layer: {name}')
    return cls(**options)
