from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..nn import CNN2Dcell, PowerModule, SeparableConvBlock

__all__ = ['FPN', 'FPNs']


class FPN(PowerModule):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
        upsample_mode: str = 'bilinear',
        use_dwconv: bool = False,
    ) -> None:
        """
        Feature Pyramid Network (FPN) module.

        Args:
            in_channels_list (List[int]):
                A list of integers representing the number of channels in each
                input feature map.
            out_channels (int):
                The number of output channels for all feature maps.
            extra_layers (int, optional):
                The number of extra down-sampling layers to add. Defaults to 0.
            out_indices (Optional[List[int]], optional):
                A list of integers indicating the indices of the feature maps to
                output. If None, all feature maps are output. Defaults to None.
            norm Optional[Union[dict, nn.Module]]:
                Optional normalization module or dictionary of its parameters.
                Defaults to None.
            act Optional[Union[dict, nn.Module]]:
                Optional activation function or dictionary of its parameters.
                Defaults to None.
            upsample_mode (str, optional):
                The type of upsampling method to use, which can be 'bilinear' or
                'nearest'. Bilinear upsampling is recommended in most cases for
                its better performance. Nearest neighbor upsampling may be useful
                when input feature maps have a small spatial resolution.
                Defaults to 'bilinear'.
            use_dwconv (bool, optional):
                Whether to use depth-wise convolution in each Conv2d block.
                Depth-wise convolution can reduce the number of parameters and
                improve computation efficiency. However, it may also degrade the
                quality of feature maps due to its low capacity.
                Defaults to False.

        Raises:
            ValueError: If the number of input feature maps does not match the length of `in_channels_list`.
                Or if `extra_layers` is negative.
        """
        super().__init__()

        self.upsample_mode = upsample_mode
        self.in_channels_list = in_channels_list

        num_in_features = len(in_channels_list)
        num_out_features = num_in_features + extra_layers

        if extra_layers < 0:
            raise ValueError('extra_layers < 0, which is not invalid.')

        conv2d = SeparableConvBlock if use_dwconv else CNN2Dcell

        self.conv1x1s = []
        for i in range(num_out_features):
            in_channels = in_channels_list[i] if i < num_in_features else in_channels_list[-1]
            if in_channels != out_channels:
                self.conv1x1s.append(
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
                self.conv1x1s.append(nn.Identity())
        self.conv1x1s = nn.ModuleList(self.conv1x1s)

        self.smooth3x3s = nn.ModuleList([
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
                    norm=getattr(nn, norm.__class__.__name__)(in_channels_list[-1]) if norm is not None else None,
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

        self.num_in_features = num_in_features
        self.out_indices = out_indices
        self.initialize_weights_()

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:

        if len(xs) != self.num_in_features:
            raise ValueError('Num of feats is not correct.')

        # make downsample if needed
        #   for example: P3, P4, P5 -> P3, P4, P5, P6, P7
        if hasattr(self, 'extra_conv_downs'):
            extras = []
            x = xs[-1]
            for conv in self.extra_conv_downs:
                x = conv(x)
                extras.append(x)
            xs = xs + extras

        # top-down pathway
        outs = []
        for i in range(len(xs)-1, -1, -1):
            out = self.conv1x1s[i](xs[i])
            if i != len(xs)-1:
                hidden = out + self.upsamples[i](hidden)
                out = self.smooth3x3s[i](hidden)
            hidden = out
            outs.append(out)
        outs = outs[::-1]

        if self.out_indices is not None:
            outs = [outs[i] for i in self.out_indices]

        return outs

    @classmethod
    def build_dwfpn(
        cls,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        upsample_mode: str = 'bilinear',
    ):
        return cls(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_layers=extra_layers,
            out_indices=out_indices,
            norm=nn.BatchNorm2d(num_features=out_channels, momentum=0.003),
            act=nn.ReLU(False),
            upsample_mode=upsample_mode,
            use_dwconv=True,
        )

    @classmethod
    def build_fpn(
        cls,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        upsample_mode: str = 'bilinear',
    ):
        return cls(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_layers=extra_layers,
            out_indices=out_indices,
            norm=nn.BatchNorm2d(num_features=out_channels, momentum=0.003),
            act=nn.ReLU(False),
            upsample_mode=upsample_mode,
            use_dwconv=False,
        )


class FPNs(PowerModule):

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        n_fpn: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        upsample_mode: str = 'bilinear',
        use_dwconv: bool = False,
    ):
        """
        Constructor of the FPN module.

        Args:

            in_channels_list (List[int]):
                A list of integers representing the number of channels in each
                input feature map.
            out_channels (int):
                The number of output channels for all feature maps.
            n_fpn (int):
                The number of FPN blocks to be stacked.
            extra_layers (int, optional):
                The number of extra down-sampling layers to add. Defaults to 0.
            out_indices (Optional[List[int]], optional):
                A list of integers indicating the indices of the feature maps to
                output. If None, all feature maps are output. Defaults to None.
            use_dwconv (bool, optional):
                Whether to use depth-wise convolution in each Conv2d block.
                Depth-wise convolution can reduce the number of parameters and
                improve computation efficiency. However, it may also degrade the
                quality of feature maps due to its low capacity.
                Defaults to False.

        Raises:
            ValueError: If the input `cls_method` is not supported.
        """
        super().__init__()
        cls_method = 'build_fpn' if not use_dwconv else 'build_dwfpn'
        num_out_features = len(in_channels_list) + extra_layers
        self.block = nn.ModuleList([
            getattr(FPN, cls_method)(
                out_channels=out_channels,
                in_channels_list=in_channels_list if i == 0 else [out_channels] * num_out_features,
                extra_layers=extra_layers if i == 0 else 0,
                out_indices=out_indices if i == n_fpn - 1 else None,
                upsample_mode=upsample_mode,
            ) for i in range(n_fpn)
        ])

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        for fpn in self.block:
            xs = fpn(xs)
        return xs
