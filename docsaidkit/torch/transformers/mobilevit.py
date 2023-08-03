from typing import List

import torch
import torch.nn as nn
from transformers import MobileViTConfig, MobileViTModel

from .utils import list_models_transformers

__all__ = ['MobileViT']


class MobileViT(nn.Module):

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 256,
        patch_size: int = 2,
        hidden_sizes: List[int] = [144, 192, 240],
        neck_hidden_sizes: List[int] = [16, 32, 64, 96, 128, 160, 640],
        num_attention_heads: int = 4,
        mlp_ratio: float = 2.0,
        expand_ratio: float = 4.0,
        hidden_act: str = "relu",
        conv_kernel_size: int = 3,
        output_stride: int = 32,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.0,
        classifier_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        qkv_bias: bool = True,
        aspp_out_channels: int = 256,
        atrous_rates: List[int] = [6, 12, 18],
        aspp_dropout_prob: float = 0.1,
        semantic_loss_ignore_index: int = 255,
        **kwargs,
    ) -> None:
        """
        This is the configuration of a `MobileViTModel`. It is used to instantiate
        a MobileViT model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield
        a similar configuration to that of the MobileViT architecture.

        [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small)

        Args:
            num_channels (int, optional):
                The number of input channels. Defaults to 3.
            image_size (int, optional):
                The size (resolution) of each image. Defaults to 256.
            patch_size (int, optional):
                The size (resolution) of each patch. Defaults to 2.
            hidden_sizes (List[int], optional):
                Dimensionality (hidden size) of the Transformer encoders at each
                stage. Defaults to [144, 192, 240]
            neck_hidden_sizes (List[int], optional):
                The number of channels for the feature maps of the backbone.
                Defaults to [16, 32, 64, 96, 128, 160, 640]
            num_attention_heads (int, optional):
                Number of attention heads for each attention layer in the
                Transformer encoder. Defaults to 4
            mlp_ratio (float, optional):
                The ratio of the number of channels in the output of the MLP to
                the number of channels in the input. Defaults to 2.0
            expand_ratio (float, optional):
                Expansion factor for the MobileNetv2 layers. Defaults to 4.0.
            hidden_act (str or function, optional):
                The non-linear activation function (function or string) in the
                Transformer encoder and convolution layers. Defaults to "relu".
            conv_kernel_size (int, optional):
                The size of the convolutional kernel in the MobileViT layer.
                Defaults to 3.
            output_stride (int, optional):
                The ratio of the spatial resolution of the output to the
                resolution of the input image. Defaults to 32.
            hidden_dropout_prob (float, optional):
                The dropout probabilitiy for all fully connected layers in the
                Transformer encoder. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional):
                The dropout ratio for the attention probabilities. Defaults to 0.0
            classifier_dropout_prob (float, optional):
                The dropout ratio for attached classifiers. Defaults to 0.1.
            initializer_range (float, optional):
                The standard deviation of the truncated_normal_initializer for
                initializing all weight matrices. Defaults to 0.02.
            layer_norm_eps (float, optional):
                The epsilon used by the layer normalization layers.
                Defaults to 1e-5.
            qkv_bias (bool, optional):
                Whether to add a bias to the queries, keys and values.
                Defaults to True.
            aspp_out_channels (int, optional):
                Number of output channels used in the ASPP layer for semantic
                segmentation. Defaults to 256.
            atrous_rates (List[int], optional):
                Dilation (atrous) factors used in the ASPP layer for semantic
                segmentation. Defaults to [6, 12, 18].
            aspp_dropout_prob (float, optional):
                The dropout ratio for the ASPP layer for semantic segmentation.
                Defaults to 0.1.
            semantic_loss_ignore_index (int, optional):
                The index that is ignored by the loss function of the semantic
                segmentation model. Defaults to 255.
        """
        super().__init__()
        self.config = MobileViTConfig(
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_sizes=hidden_sizes,
            neck_hidden_sizes=neck_hidden_sizes,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            expand_ratio=expand_ratio,
            hidden_act=hidden_act,
            conv_kernel_size=conv_kernel_size,
            output_stride=output_stride,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout_prob=classifier_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            qkv_bias=qkv_bias,
            aspp_out_channels=aspp_out_channels,
            atrous_rates=atrous_rates,
            aspp_dropout_prob=aspp_dropout_prob,
            semantic_loss_ignore_index=semantic_loss_ignore_index,
            **kwargs,
        )
        self.model = MobileViTModel(self.config, expand_output=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, all_hidden_state = self.model(x, output_hidden_states=True, return_dict=False)
        return all_hidden_state

    @staticmethod
    def list_models(author='apple', search='mobilevit') -> List[str]:
        return list_models_transformers(author=author, search=search)

    @classmethod
    def from_pretrained(cls, name, **kwargs) -> 'MobileViT':
        model = cls(**kwargs)
        kwargs.update({'expand_output': False})
        model.model = MobileViTModel.from_pretrained(name, **kwargs)
        return model
