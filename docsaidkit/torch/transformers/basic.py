import math
from typing import Tuple, Union

import torch
import torch.nn as nn

from ..nn.components import build_activation
from .token_mixer import SelfAttention

__all__ = ['ImageEncoder', 'ImageEncoderLayer']


class ImageEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        expand_ratio: float = 2,
        norm_first: bool = True,
        inner_act: Union[dict, nn.Module] = {'name': 'StarReLU'},
    ) -> None:
        """
        Initializes the EncoderLayer.

        Args:
            d_model (int):
                The number of input dimensions.
            nhead (int):
                The number of attention heads.
            expand_ratio (float, optional):
                The expansion ratio for the hidden dimensions.
                Defaults to 2.
            norm_first (bool, optional):
                Whether to apply the normalization before the attention layer.
                Defaults to True.
            inner_act (Union[dict, nn.Module], optional):
                The activation function to use for the inner feedforward layer.
                Defaults to {'name': 'StarReLU'}.
        """
        super().__init__()
        hidden_dims = int(d_model * expand_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dims),
            inner_act if isinstance(
                inner_act, nn.Module) else build_activation(**inner_act),
            nn.Linear(hidden_dims, d_model),
        )
        self.att = SelfAttention(embed_dim=d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            norm_x = self.norm1(x)
            att, att_weights = self.att(norm_x, norm_x, norm_x)
            x = x + att
            x = x + self.ffn(self.norm2(x))
        else:
            att, att_weights = self.att(x, x, x)
            x = self.norm1(x + att)
            x = self.norm2(x + self.ffn(x))
        return x, att_weights


class ImageEncoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_c: int = 3,
        *args, **kwargs,
    ) -> None:
        """
        Initialize a ImageEncoder module.

        Args:
            d_model (int):
                The input dimension of the encoder.
            num_layers (int):
                The number of layers in the encoder.
            image_size (Union[int, Tuple[int, int]]):
                The input image size.
            patch_size (Union[int, Tuple[int, int]], optional):
                The patch size. Defaults to 16.
            in_c (int):
                The number of input channels. Defaults to 3.
        """
        super().__init__()
        h, w = image_size if isinstance(
            image_size, (tuple, list)) else (image_size, image_size)
        ph, pw = patch_size if isinstance(
            patch_size, (tuple, list)) else (patch_size, patch_size)
        nh, nw = h // ph, w // pw

        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.Tensor(1, nh*nw, d_model))
        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        self.tokenizer = nn.Conv2d(
            in_c, d_model, (ph, pw), (ph, pw), bias=False)
        self.encoder = nn.ModuleList([
            ImageEncoderLayer(d_model, *args, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, cls_token: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the ImageEncoder.
        """
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_emb.expand(x.size(0), -1, -1)
        if cls_token is None:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        att_weights = []
        for layer in self.encoder:
            x, _att_weights = layer(x)
            att_weights.append(_att_weights)
        cls_token, hidden = torch.split(x, (1, x.size(1)-1), dim=1)
        return cls_token.squeeze(1), hidden, att_weights
