from typing import List, Tuple, Union

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

from .utils import list_models_transformers

__all__ = ['ViT']


class ViT(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = 'relu',
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        encoder_stride: int = 16,
        **kwargs,
    ) -> None:
        """
        ViT: Vision Transformer
        A transformer model for image classification

        Args:
            hidden_size (int, optional):
                Dimensionality of the encoder layers and the pooler layer.
                Default is 768.
            num_hidden_layers (int, optional):
                Number of hidden layers in the Transformer encoder.
                Default is 12.
            num_attention_heads (int, optional):
                Number of attention heads for each attention layer in the
                Transformer encoder.
                Default is 12.
            intermediate_size (int, optional):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer
                in the Transformer encoder.
                Default is 3072.
            hidden_act (str, optional):
                The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "selu" and "gelu_new"
                are supported.
                Default is "relu".
            hidden_dropout_prob (float, optional):
                The dropout probability for all fully connected layers in the
                embeddings, encoder, and pooler.
                Default is 0.0.
            attention_probs_dropout_prob (float, optional):
                The dropout ratio for the attention probabilities.
                Default is 0.0.
            initializer_range (float, optional):
                The standard deviation of the truncated_normal_initializer for
                initializing all weight matrices.
                Default is 0.02.
            layer_norm_eps (float, optional):
                The epsilon used by the layer normalization layers.
                Default is 1e-12.
            image_size (Union[int, Tuple[int, int]], optional):
                The size (resolution) of each image.
                Default is 224.
            patch_size (Union[int, Tuple[int, int]], optional):
                The size (resolution) of each patch.
                Default is 16.
            num_channels (int, optional):
                The number of input channels.
                Default is 3.
            qkv_bias (bool, optional):
                Whether to add a bias to the queries, keys and values.
                Default is True.
            encoder_stride (int, optional):
                Factor to increase the spatial resolution by in the decoder head
                for masked image modeling.
                Default is 16.
        """
        super().__init__()
        self.config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            qkv_bias=qkv_bias,
            encoder_stride=encoder_stride,
            **kwargs,
        )
        self.model = ViTModel(self.config, add_pooling_layer=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = self.model(x).last_hidden_state
        cls_token, hidden_state = torch.split(hidden_state, [1, hidden_state.shape[1] - 1], dim=1)
        return cls_token.squeeze(dim=1), hidden_state

    @staticmethod
    def list_models(author='google', search='vit') -> List[str]:
        return list_models_transformers(author=author, search=search)

    @classmethod
    def from_pretrained(cls, name, **kwargs) -> 'ViT':
        model = cls(**kwargs)
        kwargs.update({'add_pooling_layer': False})
        model.model = ViTModel.from_pretrained(name, **kwargs)
        return model
