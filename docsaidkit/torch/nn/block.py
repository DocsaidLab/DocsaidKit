from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .components import build_activation, build_norm
from .utils import PowerModule

__all__ = ['SeparableConvBlock']


class SeparableConvBlock(PowerModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: Optional[bool] = None,
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
    ):
        """
        A separable convolution block consisting of a depthwise convolution and a pointwise convolution.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int, optional):
                Number of output channels. If not provided, defaults to `in_channels`.
            kernel (int or Tuple[int, int], optional):
                Size of the convolution kernel. Defaults to 3.
            stride (int or Tuple[int, int], optional):
                Stride of the convolution. Defaults to 1.
            padding (int or Tuple[int, int], optional):
                Padding added to all four sides of the input. Defaults to 1.
            bias (bool, optional):
                Whether to include a bias term in the convolutional layer.
                If bias = None, bias would be set as Ture when normalization layer is None and
                False when normalization layer is not None.
                Defaults to None.
            norm (dict or nn.Module, optional):
                Configuration of normalization layer. Defaults to None.
            act (dict or nn.Module, optional):
                Configuration of activation layer. Defaults to None.
        """
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        if bias is None:
            bias = True if norm is None else False

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm = build_norm(**norm) if isinstance(norm, dict) else norm
        self.act = build_activation(**act) if isinstance(act, dict) else act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x
