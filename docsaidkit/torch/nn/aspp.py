import torch
import torch.nn as nn

from .cnn import CNN2Dcell
from .components import Hswish
from .utils import PowerModule

__all__ = ['ASPPLayer']

__doc__ = """
    REFERENCES: DeepLab: Semantic Image Segmentation with Deep Convolutional
                Nets, Atrous Convolution, and Fully Connected CRFs
    URL: https://arxiv.org/pdf/1606.00915.pdf
"""


class ASPPLayer(PowerModule):

    ARCHS = {
        # ksize, stride, padding, dilation, is_use_hs
        'DILATE1': [3, 1, 1, 1, True],
        'DILATE2': [3, 1, 2, 2, True],
        'DILATE3': [3, 1, 4, 4, True],
        'DILATE4': [3, 1, 8, 8, True],
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_activate: nn.Module = nn.ReLU(),
    ):
        """
        Constructor for the ASPPLayer class.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            output_activate (nn.Module, optional):
                Activation function to apply to the output. Defaults to nn.ReLU().
        """
        super().__init__()
        self.layers = nn.ModuleDict()
        for dilate_name, cfg in self.ARCHS.items():
            ksize, stride, padding, dilation, use_hs = cfg
            layer = CNN2Dcell(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                norm=nn.BatchNorm2d(in_channels),
                act=Hswish() if use_hs else nn.ReLU(),
            )
            self.layers[dilate_name] = layer

        self.output_layer = CNN2Dcell(
            in_channels=in_channels * len(self.layers),
            out_channels=out_channels,
            kernel=1,
            stride=1,
            padding=0,
            norm=nn.BatchNorm2d(out_channels),
            act=output_activate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [layer(x) for layer in self.layers.values()]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.output_layer(outputs)
        return outputs
