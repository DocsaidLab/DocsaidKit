from typing import List

import torch
from timm import create_model

from ..neck import BiFPNs
from ..nn import PowerModule

__all__ = ['EfficientDet']


class EfficientDet(PowerModule):

    def __init__(self, compound_coef: int = 0, pretrained: bool = True, **kwargs):
        """
        EfficientDet backbone.

        Args:
            compound_coef (int, optional):
                Compound scaling factor for the model architecture. Defaults to 0.
            pretrained (bool, optional):
                If True, returns a model pre-trained on ImageNet. Defaults to True.
        """
        super().__init__()
        self.compound_coef = compound_coef

        # Number of filters for each FPN layer at each compound coefficient
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]

        # Number of BiFPN repeats for each compound coefficient
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]

        # Number of channels for each input feature map at each compound coefficient
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [80, 224, 640],
            8: [88, 248, 704],
        }

        self.backbone = create_model(
            f'efficientnet_b{compound_coef}',
            pretrained=pretrained,
            features_only=True,
            exportable=True,
        )

        self.bifpn = BiFPNs(
            in_channels_list=conv_channel_coef[compound_coef],
            out_channels=self.fpn_num_filters[compound_coef],
            n_bifpn=self.fpn_cell_repeats[compound_coef],
            attention=True if compound_coef < 6 else False,
            extra_layers=3 if compound_coef > 7 else 2,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the EfficientDet backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            List[torch.Tensor]: A list of feature maps, each with shape (batch_size, channels, height, width),
                                where the number of feature maps is equal to the number of BiFPN layers.
        """
        return self.bifpn(self.backbone(x)[2:])
