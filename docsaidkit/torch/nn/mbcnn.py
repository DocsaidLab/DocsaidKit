from copy import deepcopy
from typing import Tuple, Union

import torch
import torch.nn as nn

from .cnn import CNN2Dcell
from .components import build_norm
from .selayer import SELayer
from .utils import PowerModule

__all__ = ['MBCNNcell']


class MBCNNcell(PowerModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: int = None,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        use_se: bool = False,
        se_reductioin: int = 4,
        inner_norm: Union[dict, nn.Module] = None,
        inner_act: Union[dict, nn.Module] = None,
        norm: Union[dict, nn.Module] = None,
    ):
        """
        This neural network block is commonly known as the "inverted residual block",
        which is used in MobileNetV2, MobileNetV3, and EfficientNet (but not always).
        ref: https://arxiv.org/pdf/1905.02244.pdf

        For MobileNetV1, the block consists of a kxk depth-wise convolution with
        group normalization, batch normalization, and ReLU activation, followed
        by a 1x1 projection with batch normalization.

        mbv1:
            input ---> kxk depth-wise (group, bn, relu) ---> 1x1 projection (bn)

        For MobileNetV2, the block starts with a 1x1 expansion with batch normalization
        and ReLU6 activation, followed by a kxk depth-wise convolution with group
        normalization, batch normalization, and ReLU6 activation, and ends with a
        1x1 projection with batch normalization.

        mbv2:
            input ---> 1x1 expansion (bn, relu6) ---> kxk depth-wise (group, bn, relu6) ---> 1x1 projection (bn)

        For MobileNetV3, the block starts with a 1x1 expansion with batch normalization
        and h-swish activation, followed by a kxk depth-wise convolution with group
        normalization, batch normalization, and h-swish activation, and ends with a
        1x1 projection with batch normalization. In addition, MobileNetV3 uses a
        squeeze-and-excitation (SE) layer to enhance feature interdependencies.

        mbv3:
            input ---> 1x1 expansion (bn, hswish) ---> kxk depth-wise (group, bn, hswish) ---> 1x1 projection (bn)
                        |                                          ↑
                        ↓---------->    SE layer (v3)     -------->|


        Args:
            in_channels (int):
                The number of input channels.
            hid_channels (int):
                The number of hidden channels for expanding dimensions.
            out_channels (int):
                The number of output channels.
            kernel (Union[int, Tuple[int, int]], optional):
                The kernel size of the depth-wise convolution. Defaults to 3.
            stride (int, optional):
                The stride size of the depth-wise convolution. Defaults to 1.
            use_se (bool, optional):
                Whether to use the SE layer. Defaults to True.
            se_reduction (int, optional):
                Reduction ratio for the number of hidden channels in the SE layer.
                Defaults to 4.
            inner_norm (Union[dict, nn.Module], optional):
                Dictionary or function that creates a normalization layer inside
                the MB block. Defaults to None.
            inner_act (Union[dict, nn.Module], optional):
                Dictionary or function that creates an activation layer inside
                the MB block. Defaults to None.
            norm (Union[dict, nn.Module], optional):
                Dictionary or function that creates a normalization layer on the
                last stage. Defaults to None.
        """
        super().__init__()
        self.identity = stride == 1 and in_channels == out_channels

        if hid_channels is None:
            hid_channels = in_channels

        if hid_channels != in_channels:
            self.expdim = CNN2Dcell(
                in_channels,
                hid_channels,
                kernel=1,
                stride=1,
                padding=0,
                norm=deepcopy(inner_norm),
                act=deepcopy(inner_act),
            )

        padding = (kernel - 1) // 2 if isinstance(kernel, int) else \
            ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)

        self.dwise = CNN2Dcell(
            hid_channels,
            hid_channels,
            kernel=kernel,
            stride=stride,
            padding=padding,
            groups=hid_channels,
            norm=deepcopy(inner_norm),
            act=deepcopy(inner_act),
        )

        if use_se:
            self.dwise_se = SELayer(
                hid_channels,
                se_reductioin,
            )

        self.pwise_linear = CNN2Dcell(
            hid_channels,
            out_channels,
            kernel=1,
            stride=1,
            padding=0,
        )

        if norm is not None:
            self.norm = norm if isinstance(norm, nn.Module) else build_norm(**norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if hasattr(self, 'expdim'):
            out = self.expdim(out)
        out = self.dwise(out)
        if hasattr(self, 'dwise_se'):
            out = self.dwise_se(out)
        out = self.pwise_linear(out)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = x + out if self.identity else out  # skip connection
        return out

    @classmethod
    def build_mbv1block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
    ):
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=in_channels,
            kernel=kernel,
            stride=stride,
            use_se=False,
            inner_norm=nn.BatchNorm2d(in_channels),
            inner_act=nn.ReLU(False),
            norm=nn.BatchNorm2d(out_channels),
        )

    @classmethod
    def build_mbv2block(
        cls,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 2,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
    ):
        hid_channels = int(in_channels * expand_ratio)
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            kernel=kernel,
            stride=stride,
            use_se=False,
            inner_norm=nn.BatchNorm2d(hid_channels),
            inner_act=nn.ReLU6(False),
            norm=nn.BatchNorm2d(out_channels),
        )

    @classmethod
    def build_mbv3block(
        cls,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 2,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
    ):
        hid_channels = int(in_channels * expand_ratio)
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            kernel=kernel,
            stride=stride,
            use_se=True,
            inner_norm=nn.BatchNorm2d(hid_channels),
            inner_act=nn.Hardswish(False),
            norm=nn.BatchNorm2d(out_channels),
        )
