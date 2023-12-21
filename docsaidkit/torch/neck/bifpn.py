from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..nn import (CNN2Dcell, PowerModule, SeparableConvBlock, WeightedSum,
                  build_activation, build_norm)

__all__ = ['BiFPN', 'BiFPNs']


class BiFPN(PowerModule):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
        upsample_mode: str = 'bilinear',
        use_conv: bool = False,
        attention: bool = True,
    ) -> None:
        """
        BiFPN (Bidirectional Feature Pyramid Network) is a type of feature extraction
        module commonly used in object detection and instance segmentation tasks.
        It was introduced in the EfficientDet paper by Mingxing Tan et al. in 2020.

        BiFPN is an extension of the FPN (Feature Pyramid Network) module that
        incorporates bidirectional connections between feature maps of different
        resolutions. It takes multiple feature maps with different spatial resolutions
        and merges them into a single feature map with consistent resolution.

        The bidirectional connections enable efficient feature propagation and
        fusion across multiple scales, improving the quality of the extracted
        features and ultimately leading to better performance in object detection
        and instance segmentation.

        BiFPN consists of several stages of repeated bifurcated networks, where
        each stage processes the output of the previous stage. The bifurcated
        network consists of a top-down path that performs upsampling, and a
        bottom-up path that performs downsampling. The outputs of both paths are
        then fused together through a learnable weight sharing mechanism. This
        process is repeated across different scales and levels of the feature
        pyramid, creating a hierarchical feature representation that captures
        rich and diverse information.

        BiFPN has become a popular choice for many state-of-the-art object detection
        and instance segmentation architectures due to its efficiency and effectiveness
        in feature extraction.

        Args:
            in_channels_list (List[int]):
                A list of integers representing the number of channels in each
                input feature map.
            out_channels (int):
                The number of output channels for all feature maps.
            extra_layers (int, optional):
                The number of extra down-sampling layers to add. Defaults to 0.
            out_indices (Optional[List[int]], optional):
                A list of integers indicating the indices of the feature maps
                to output. If None, all feature maps are output. Defaults to None.
            norm Optional[Union[dict, nn.Module]]:
                Optional normalization module or dictionary of its parameters.
                Defaults to None.
            act Optional[Union[dict, nn.Module]]:
                Optional activation function or dictionary of its parameters.
                Defaults to None.
            upsample_mode (str, optional):
                The type of upsampling method to use, which can be 'bilinear'
                or 'nearest'. Bilinear upsampling is recommended in most cases
                for its better performance. Nearest neighbor upsampling may be
                useful when input feature maps have a small spatial resolution.
                Defaults to 'bilinear'.
            use_conv (bool, optional):
                In BiFPN, SeparableConvBlock is used by default to replace CNN.
                If you want to use a general CNN, set use_conv to True.
                Defaults to False.
            attention (bool, optional):
                Whether to use attention mechanism in each WeightedSum block.

        Raises:
            ValueError: If the number of input feature maps does not match the
            length of `in_channels_list` or if `extra_layers` is negative.
        """
        super().__init__()

        self.attention = attention
        self.upsample_mode = upsample_mode
        self.in_channels_list = in_channels_list

        num_in_features = len(in_channels_list)
        num_out_features = num_in_features + extra_layers

        if extra_layers < 0:
            raise ValueError('extra_layers < 0, which is not invalid.')

        conv2d = CNN2Dcell if use_conv else SeparableConvBlock

        # Lateral layers
        conv1x1s = []
        for i in range(num_out_features):
            in_channels = in_channels_list[i] if i < num_in_features else in_channels_list[-1]
            if in_channels != out_channels:
                conv1x1s.append(
                    CNN2Dcell(
                        in_channels,
                        out_channels,
                        kernel=1,
                        stride=1,
                        padding=0,
                        norm=deepcopy(norm),
                    )
                )
            else:
                conv1x1s.append(nn.Identity())
        self.conv1x1s = nn.ModuleList(conv1x1s)

        self.conv_up_3x3s = nn.ModuleList([
            conv2d(
                out_channels,
                out_channels,
                kernel=3,
                stride=1,
                padding=1,
                norm=deepcopy(norm),
                act=deepcopy(act),
            )
            for _ in range(num_out_features - 1)
        ])

        self.conv_down_3x3s = nn.ModuleList([
            conv2d(
                out_channels,
                out_channels,
                kernel=3,
                stride=1,
                padding=1,
                norm=deepcopy(norm),
                act=deepcopy(act),
            )
            for _ in range(num_out_features - 1)
        ])

        if extra_layers > 0:
            self.extra_conv_downs = nn.ModuleList([
                conv2d(
                    in_channels_list[-1],
                    in_channels_list[-1],
                    kernel=3,
                    stride=2,
                    padding=1,
                    norm=nn.BatchNorm2d(
                        in_channels_list[-1]) if norm is not None else None,
                    act=deepcopy(act),
                )
                for _ in range(extra_layers)
            ])

        self.upsamples = nn.ModuleList([
            nn.Upsample(
                scale_factor=2,
                mode=upsample_mode,
                align_corners=False if upsample_mode != 'nearest' else None,
            )
            for _ in range(num_out_features - 1)
        ])

        self.downsamples = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )
            for _ in range(num_out_features - 1)
        ])

        # Weight
        self.weighted_sum_2_input = nn.ModuleList([
            WeightedSum(2, act=nn.ReLU(False), requires_grad=attention)
            for _ in range(num_out_features)
        ])

        self.weighted_sum_3_input = nn.ModuleList([
            WeightedSum(3, act=nn.ReLU(False), requires_grad=attention)
            for _ in range(num_out_features-2)
        ])

        self.num_in_features = num_in_features
        self.out_indices = out_indices
        self.initialize_weights_()

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        if len(xs) != self.num_in_features:
            raise ValueError(
                'The length of given xs is not equal to the length of in_channels_list.'
            )

        # make extra levels by conv2d if needed
        #   for example: P3, P4, P5 -> P3, P4, P5, P6, P7
        if hasattr(self, 'extra_conv_downs'):
            extras = []
            x = xs[-1]
            for conv in self.extra_conv_downs:
                x = conv(x)
                extras.append(x)
            xs = xs + extras

        # Fixed input channels
        out_fixed = [self.conv1x1s[i](xs[i]) for i in range(len(xs))]

        # top-down pathway
        outs_top_down = []
        for i in range(len(out_fixed)-1, -1, -1):
            out = out_fixed[i]
            if i != len(xs)-1:
                hidden = self.weighted_sum_2_input[i](
                    [out, self.upsamples[i](hidden)])
                out = self.conv_up_3x3s[i](hidden)
            hidden = out
            outs_top_down.append(out)
        outs_top_down = outs_top_down[::-1]

        # down-top pathway
        outs_down_top = []
        for i in range(len(outs_top_down)):
            out = outs_top_down[i]
            residual = out_fixed[i]
            if i != 0 and i != len(outs_top_down) - 1:
                hidden = self.weighted_sum_3_input[i - 1](
                    [out, self.downsamples[i - 1](hidden), residual])
                out = self.conv_down_3x3s[i - 1](hidden)
            elif i == len(outs_top_down) - 1:
                hidden = self.weighted_sum_2_input[0](
                    [self.downsamples[i - 1](hidden), residual])
                out = self.conv_down_3x3s[i - 1](hidden)

            hidden = out
            outs_down_top.append(out)

        if self.out_indices is not None:
            outs_down_top = [outs_down_top[i] for i in self.out_indices]

        return outs_down_top

    @classmethod
    def build_convbifpn(
        cls,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        upsample_mode: str = 'bilinear',
        attention: bool = True,
    ):
        return cls(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_layers=extra_layers,
            out_indices=out_indices,
            norm=nn.BatchNorm2d(num_features=out_channels,
                                momentum=0.003, eps=1e-4),
            act=nn.ReLU(False),
            upsample_mode=upsample_mode,
            use_conv=True,
            attention=attention,
        )

    @classmethod
    def build_bifpn(
        cls,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        upsample_mode: str = 'bilinear',
        attention: bool = True,
    ):
        return cls(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_layers=extra_layers,
            out_indices=out_indices,
            norm=nn.BatchNorm2d(num_features=out_channels,
                                momentum=0.003, eps=1e-4),
            act=nn.ReLU(False),
            upsample_mode=upsample_mode,
            use_conv=False,
            attention=attention,
        )


class BiFPNs(PowerModule):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        n_bifpn: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        upsample_mode: str = 'bilinear',
        attention: bool = True,
        use_conv: bool = False,
    ):
        """
        Constructor of the BiFPN module.

        Args:
            in_channels_list (List[int]):
                A list of integers representing the number of input channels for
                each feature map.
            out_channels (int):
                The number of output channels for each feature map.
            n_bifpn (int):
                The number of BiFPN blocks to be stacked.
            extra_layers (int, optional):
                The number of additional convolutional layers added after the
                BiFPN blocks. Defaults to 0.
            out_indices (Optional[List[int]], optional):
                A list of integers representing the indices of output feature maps.
                Defaults to None.
            upsample_mode (str, optional):
                The interpolation method used in the upsampling operation.
                Defaults to 'bilinear'.
            attention (bool, optional):
                A boolean flag indicating whether to use attention mechanism.
                Defaults to True.
            use_conv (bool, optional):
                In BiFPN, SeparableConvBlock is used by default to replace CNN.
                If you want to use a general CNN, set use_conv to True.
                Defaults to False.

        Raises:
            ValueError: If the input `cls_method` is not supported.
        """
        super().__init__()
        cls_method = 'build_bifpn' if not use_conv else 'build_convbifpn'
        num_out_features = len(in_channels_list) + extra_layers
        self.block = nn.ModuleList([
            getattr(BiFPN, cls_method)(
                out_channels=out_channels,
                in_channels_list=in_channels_list if i == 0 else [
                    out_channels] * num_out_features,
                extra_layers=extra_layers if i == 0 else 0,
                out_indices=out_indices if i == n_bifpn - 1 else None,
                attention=attention,
                upsample_mode=upsample_mode,
            ) for i in range(n_bifpn)
        ])

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        for bifpn in self.block:
            xs = bifpn(xs)
        return xs
