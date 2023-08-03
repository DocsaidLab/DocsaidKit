from typing import Tuple, Union

import torch
import torch.nn as nn

from ..nn.components import LayerNorm2d, build_activation
from ..nn.mbcnn import MBCNNcell

__all__ = [
    'Attention', 'AttentionMixing', 'RandomMixing', 'SepConvMixing',
    'PoolMixing',
]


class Attention(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        return_attn: bool = False,
        add_output_layer: bool = True,
        is_cross_attention: bool = False,
    ) -> None:
        """
        Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
        Modified from timm.

        Args:
            dim (int):
                Number of input channels.
            head_dim (int, optional):
                Dimensionality of the output of each head, defaults to 32.
            num_heads (int, optional):
                Number of attention heads, defaults to None (uses `dim` divided by `head_dim`).
            qkv_bias (bool, optional):
                Whether to include bias in the projection layers, defaults to False.
            return_attn (bool, optional):
                Whether to return the attention map, defaults to False.
            add_output_layer (bool, optional):
                Whether to add an output layer, defaults to True.
            is_cross_attention (bool, optional):
                Whether this is cross-attention, defaults to False.
        """
        super().__init__()
        assert in_features % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5
        self.return_attn = return_attn
        self.is_cross_attention = is_cross_attention

        self.proj = nn.Linear(in_features, in_features) \
            if add_output_layer else nn.Identity()

        if self.is_cross_attention:
            self.q = nn.Linear(in_features, in_features, bias=qkv_bias)
            self.kv = nn.Linear(in_features, in_features * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(in_features, in_features * 3, bias=qkv_bias)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> torch.Tensor:
        """
        Applies self-attention to the input tensor.

        Args:
            x:
                Input tensor of shape (batch_size, seq_len, dim).
            hidden_state:
                Hidden state of the previous token, used for cross-attention.

        Returns:
            A tuple containing the output tensor of shape (batch_size, seq_len, dim) and
            the attention tensor of shape (batch_size, num_heads, seq_len, seq_len).
        """
        B, N, C = x.shape

        if self.is_cross_attention:
            q = self.q(x)
            kv = self.kv(hidden_state)
            k, v = torch.chunk(kv, 2, dim=-1)
        else:
            qkv = self.qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if self.return_attn:
            return x, attn

        return x


class AttentionMixing(Attention):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies self-attention to the input tensor.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            A tensor of the same shape as input after applying the self-attention.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = super().forward(x)
        if self.return_attn:
            x, attn = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        if self.return_attn:
            return x, attn
        return x


class RandomMixing(nn.Module):

    def __init__(self, num_tokens: int):
        """ Random mixing of tokens.
        Args:
            num_tokens (int):
                Number of tokens.
        """
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        """
        Applies random-attention to the input tensor.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            A tensor of the same shape as input after applying the random-attention.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = torch.einsum('mn, bcn -> bcm', self.random_matrix, x)
        x = x.reshape(B, C, H, W)
        return x


class SepConvMixing(nn.Module):

    def __init__(
        self,
        in_features: int,
        expand_ratio: float = 2,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        inner_act: Union[dict, nn.Module] = {'name': 'StarReLU'},
    ) -> None:
        """
        SepConvMixing is an inverted separable convolution block from MobileNetV2.
        It performs a depthwise convolution followed by a pointwise convolution.
        Ref: https://arxiv.org/abs/1801.04381.

        Args:
            in_channels (int):
                Number of input channels.
            expand_ratio (float):
                Expansion ratio of the hidden channels. Defaults to 2.
            kernel_size (Union[int, Tuple[int, int]]):
                Size of the depthwise convolution kernel. Defaults to 7.
            inner_act (Union[dict, nn.Module]):
                Activation function to be used internally. Defaults to StarReLU.
        """
        super().__init__()
        hid_channels = int(in_features * expand_ratio)
        self.mbcnn_v2 = MBCNNcell(
            in_channels=in_features,
            out_channels=in_features,
            hid_channels=hid_channels,
            kernel=kernel_size,
            norm=LayerNorm2d(in_features),
            inner_norm=LayerNorm2d(hid_channels),
            inner_act=inner_act if isinstance(inner_act, nn.Module) else build_activation(**inner_act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a SepConvMixing operation on the input tensor.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            A tensor of the same shape as the input after applying mbcnn_v2 module.
        """
        return self.mbcnn_v2(x)


class PoolMixing(nn.Module):

    def __init__(self, pool_size: int = 3):
        """
        Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418

        Args:
            pool_size (int): Size of the pooling window.
        """
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size//2,
            count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling and subtract the result from the input.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            A tensor of the same shape as input after applying the pooling and subtraction.
        """
        return self.pool(x) - x
