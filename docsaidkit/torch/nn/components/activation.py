from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import (CELU, ELU, GELU, GLU, Hardsigmoid,
                                         Hardswish, Hardtanh, LeakyReLU,
                                         LogSigmoid, LogSoftmax,
                                         MultiheadAttention, PReLU, ReLU,
                                         ReLU6, RReLU, Sigmoid, SiLU, Softmax,
                                         Softmax2d, Softmin, Softplus,
                                         Softshrink, Softsign, Tanh,
                                         Tanhshrink, Threshold)

__all__ = [
    'Swish', 'Hsigmoid', 'Hswish', 'build_activation', 'StarReLU', 'SquaredReLU',
]

__all__ += ['CELU', 'ELU', 'GELU', 'GLU', 'LeakyReLU', 'LogSigmoid',
            'LogSoftmax', 'MultiheadAttention', 'PReLU', 'ReLU', 'ReLU6',
            'RReLU', 'Sigmoid', 'Softmax', 'Softmax2d', 'Softmin', 'Softplus',
            'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold',
            'Hardsigmoid', 'Hardswish', 'Hardtanh', 'SiLU',]


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        return F.relu6(x + 3., inplace=self.inplace) * 0.16666666667


class Hswish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        return x * F.relu6(x + 3., inplace=self.inplace) * 0.16666666667


class StarReLU(nn.Module):

    def __init__(
        self,
        scale: float = 1.0,
        bias: float = 0.0,
        scale_learnable: bool = True,
        bias_learnable: bool = True,
        inplace: bool = False
    ):
        """
        StarReLU: s * relu(x) ** 2 + b
        Ref: MetaFormer Baselines for Vision (2022.12) (https://arxiv.org/pdf/2210.13452.pdf)

        Args:
            scale (float):
                Scale factor for the activation function, defaults to 1.0.
            bias (float):
                Bias for the activation function, defaults to 0.0.
            scale_learnable (bool):
                Whether the scale factor should be learnable, defaults to True.
            bias_learnable (bool):
                Whether the bias should be learnable, defaults to True.
            inplace (bool):
                Whether to modify the input in place, defaults to False.
        """
        super().__init__()
        self.inplace = inplace
        self.scale = nn.Parameter(
            torch.tensor(scale),
            requires_grad=scale_learnable
        )
        self.bias = nn.Parameter(
            torch.tensor(bias),
            requires_grad=bias_learnable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.relu(x, inplace=self.inplace) ** 2 + self.bias


class SquaredReLU(nn.Module):

    def __init__(self, inplace=False):
        """ Squared ReLU: https://arxiv.org/abs/2109.08668 """
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.square(F.relu(x, inplace=self.inplace))


# Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
Swish = nn.SiLU


def build_activation(name, **options) -> Union[nn.Module, None]:
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'Activation named {name} is not supported.')
    return cls(**options)
