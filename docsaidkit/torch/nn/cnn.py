from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .components import build_activation, build_dropout, build_norm, build_pool
from .utils import PowerModule

__all__ = [
    'CNN2Dcell',
]


class CNN2Dcell(PowerModule):

    def __init__(
        self,
        in_channels: Union[float, int],
        out_channels: Union[float, int],
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        padding_mode: str = 'zeros',
        norm: Union[dict, nn.Module] = None,
        dropout: Union[dict, nn.Module] = None,
        act: Union[dict, nn.Module] = None,
        pool: Union[dict, nn.Module] = None,
        init_type: str = 'normal',
    ):
        """
        This class is used to build a 2D convolutional neural network cell.

        Args:
            in_channels (int or float):
                Number of input channels.
            out_channels (int or float):
                Number of output channels.
            kernel (int or tuple, optional):
                Size of the convolutional kernel. Defaults to 3.
            stride (int or tuple, optional):
                Stride size. Defaults to 1.
            padding (int or tuple, optional):
                Padding size. Defaults to 1.
            dilation (int, optional):
                Spacing between kernel elements. Defaults to 1.
            groups (int, optional):
                Number of blocked connections from input channels to output
                channels. Defaults to 1.
            bias (bool, optional):
                Whether to include a bias term in the convolutional layer.
                If bias = None, bias would be set as Ture when normalization layer is None and
                False when normalization layer is not None.
                Defaults to None.
            padding_mode (str, optional):
                Options = {'zeros', 'reflect', 'replicate', 'circular'}.
                Defaults to 'zeros'.
            norm (Union[dict, nn.Module], optional):
                normalization layer or a dictionary of arguments for building a
                normalization layer. Default to None.
            dropout (Union[dict, nn.Module], optional):
                dropout layer or a dictionary of arguments for building a dropout
                layer. Default to None.
            act (Union[dict, nn.Module], optional):
                Activation function or a dictionary of arguments for building an
                activation function. Default to None.
            pool (Union[dict, nn.Module], optional):
                pooling layer or a dictionary of arguments for building a pooling
                layer. Default to None.
            init_type (str):
                Method for initializing model parameters. Default to 'normal'.
                Options = {'normal', 'uniform'}

        Examples for using norm, act, and pool:
            1. cell = CNN2Dcell(in_channels=3,
                                out_channels=12,
                                norm=nn.BatchNorm2d(12),
                                act=nn.ReLU(),
                                pool=nn.AdaptiveAvgPool2d(1))
            2. cell = CNN2Dcell(in_channels=3,
                                out_channels=12,
                                norm={'name': 'BatchNorm2d', 'num_features': 12},
                                act={'name': 'ReLU', 'inplace': True})

        Attributes:
            layer (nn.ModuleDict): a dictionary of layer contained in the cell.
        """
        super().__init__()
        self.layer = nn.ModuleDict()

        if bias is None:
            bias = True if norm is None else False

        self.layer['cnn'] = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        optional_modules = OrderedDict({
            'norm': build_norm(**norm) if isinstance(norm, dict) else norm,
            'dp': build_dropout(**dropout) if isinstance(dropout, dict) else dropout,
            'act': build_activation(**act) if isinstance(act, dict) else act,
            'pool': build_pool(**pool) if isinstance(pool, dict) else pool,
        })
        for name, m in optional_modules.items():
            if m is not None:
                self.layer[name] = m
        self.initialize_weights_(init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, m in self.layer.items():
            x = m(x)
        return x
