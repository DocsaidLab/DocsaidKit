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
        n_dims: int,
        expand_ratio: float = 2,
        norm_first: bool = True,
        inner_act: Union[dict, nn.Module] = {'name': 'StarReLU'},
    ) -> None:
        """
        Initializes the EncoderLayer.

        Args:
            n_dims (int):
                The number of input dimensions.
            expand_ratio (float, optional):
                The expansion ratio for the hidden dimensions.
                Defaults to 2.
            norm_first (bool, optional):
                Whether to apply the normalization before the attention layer.
                Defaults to True.
            add_attention_output_layer (bool, optional):
                Whether to add an output layer to the attention module.
                Defaults to True.
        """
        super().__init__()
        hidden_dims = int(n_dims * expand_ratio)
        self.fc = nn.Sequential(
            nn.Linear(n_dims, hidden_dims),
            inner_act if isinstance(
                inner_act, nn.Module) else build_activation(**inner_act),
            nn.Linear(hidden_dims, n_dims),
        )
        self.att = SelfAttention(embed_dim=n_dims)
        self.norm1 = nn.LayerNorm(n_dims)
        self.norm2 = nn.LayerNorm(n_dims)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = x + self.att(self.norm1(x))
            x = x + self.fc(self.norm2(x))
        else:
            x = self.norm1(x + self.att(x))
            x = self.norm2(x + self.fc(x))
        return x


class ImageEncoder(nn.Module):

    def __init__(
        self,
        n_dims: int,
        n_layers: int,
        num_patches: Union[int, Tuple[int, int]] = None,
        *args, **kwargs,
    ) -> None:
        """
        Initialize a ImageEncoder module.

        Args:
            n_dims (int):
                The input dimension of the encoder.
            n_layers (int):
                The number of layers in the encoder.
            norm_first (bool, optional):
                Whether to apply the normalization before the attention layer.
                Defaults to True.
            num_patches (Union[int, Tuple[int, int]], optional):
                The number of patches that can fit into the image. if None,
                model will not use positional embeddings.
                Defaults to None.
        """
        super().__init__()
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, n_dims))
        self.pos_emb = nn.Parameter(torch.Tensor(num_patches, 1, n_dims))

        nn.init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb, a=math.sqrt(5))

        self.encoder = nn.Sequential(*[
            ImageEncoderLayer(n_dims, *args, **kwargs)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ImageEncoder module with image input.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output of the
            classification token and the output of the transformer encoder layers.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        x = x + self.pos_emb
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.encoder(x)
        cls_token, hidden = torch.split(x, (1, H*W), dim=1)
        return cls_token.squeeze(1), hidden
