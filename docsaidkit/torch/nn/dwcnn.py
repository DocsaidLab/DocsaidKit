from collections import OrderedDict

import torch.nn as nn

__all__ = ['depth_conv2d', 'conv_dw', 'conv_dw_in']


def depth_conv2d(in_channels: int, out_channels: int, kernel: int = 1, stride: int = 1, pad: int = 0):
    return nn.Sequential(
        OrderedDict([
            ('conv3x3', nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=pad, groups=in_channels),),
            ('act', nn.ReLU(),),
            ('conv1x1', nn.Conv2d(in_channels, out_channels, kernel_size=1)),
        ])
    )


def conv_dw(in_channels: int, out_channels: int, stride: int, act: nn.Module = nn.ReLU()):
    return nn.Sequential(
        OrderedDict([
            ('conv3x3', nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)),
            ('bn1', nn.BatchNorm2d(in_channels)),
            ('act1', nn.ReLU()),
            ('conv1x1', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('act2', act),
        ])
    )


def conv_dw_in(in_channels: int, out_channels: int, stride: int, act: nn.Module = nn.ReLU()):
    return nn.Sequential(
        OrderedDict([
            ('conv3x3', nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)),
            ('in1', nn.InstanceNorm2d(in_channels)),
            ('act1', nn.ReLU()),
            ('conv1x1', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
            ('in2', nn.InstanceNorm2d(out_channels)),
            ('act2', act),
        ])
    )
