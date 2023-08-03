from typing import List

import torch
import torch.nn as nn
from transformers import EfficientFormerConfig, EfficientFormerModel

from .utils import list_models_transformers

__all__ = ['EfficientFormer']


class EfficientFormer(nn.Module):

    def __init__(
        self,
        depths: List[int] = [3, 2, 6, 4],
        hidden_sizes: List[int] = [48, 96, 224, 448],
        downsamples: List[bool] = [True, True, True, True],
        dim: int = 448,
        key_dim: int = 32,
        attention_ratio: int = 4,
        resolution: int = 7,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 8,
        mlp_expansion_ratio: int = 4,
        hidden_dropout_prob: float = 0.0,
        patch_size: int = 16,
        num_channels: int = 3,
        pool_size: int = 3,
        downsample_patch_size: int = 3,
        downsample_stride: int = 2,
        downsample_pad: int = 1,
        drop_path_rate: float = 0.0,
        num_meta3d_blocks: int = 1,
        distillation: bool = True,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ) -> None:
        r"""
        This is the configuration class to store the configuration of an
        [`EfficientFormerModel`]. It is used to instantiate an EfficientFormer
        model according to the specified arguments, defining the model architecture.

        Instantiating a configuration with the defaults will yield a similar
        configuration to that of the EfficientFormer [snap-research/efficientformer-l1]
        (https://huggingface.co/snap-research/efficientformer-l1) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used
        to control the model outputs. Read the documentation from [`PretrainedConfig`]
        for more information.

        Args:
            depths (`List(int)`, *optional*, defaults to `[3, 2, 6, 4]`)
                Depth of each stage.
            hidden_sizes (`List(int)`, *optional*, defaults to `[48, 96, 224, 448]`)
                Dimensionality of each stage.
            downsamples (`List(bool)`, *optional*, defaults to `[True, True, True, True]`)
                Whether or not to downsample inputs between two stages.
            dim (`int`, *optional*, defaults to 448):
                Number of channels in Meta3D layers
            key_dim (`int`, *optional*, defaults to 32):
                The size of the key in meta3D block.
            attention_ratio (`int`, *optional*, defaults to 4):
                Ratio of the dimension of the query and value to the dimension
                of the key in MSHA block
            resolution (`int`, *optional*, defaults to 5)
                Size of each patch
            num_hidden_layers (`int`, *optional*, defaults to 5):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 8):
                Number of attention heads for each attention layer in the 3D
                MetaBlock.
            mlp_expansion_ratio (`int`, *optional*, defaults to 4):
                Ratio of size of the hidden dimensionality of an MLP to the
                dimensionality of its input.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the
                embeddings and encoder.
            patch_size (`int`, *optional*, defaults to 16):
                The size (resolution) of each patch.
            num_channels (`int`, *optional*, defaults to 3):
                The number of input channels.
            pool_size (`int`, *optional*, defaults to 3):
                Kernel size of pooling layers.
            downsample_patch_size (`int`, *optional*, defaults to 3):
                The size of patches in downsampling layers.
            downsample_stride (`int`, *optional*, defaults to 2):
                The stride of convolution kernels in downsampling layers.
            downsample_pad (`int`, *optional*, defaults to 1):
                Padding in downsampling layers.
            drop_path_rate (`int`, *optional*, defaults to 0):
                Rate at which to increase dropout probability in DropPath.
            num_meta3d_blocks (`int`, *optional*, defaults to 1):
                The number of 3D MetaBlocks in the last stage.
            distillation (`bool`, *optional*, defaults to `True`):
                Whether to add a distillation head.
            use_layer_scale (`bool`, *optional*, defaults to `True`):
                Whether to scale outputs from token mixers.
            layer_scale_init_value (`float`, *optional*, defaults to 1e-5):
                Factor by which outputs from token mixers are scaled.
            hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the
                encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and
                `"gelu_new"` are supported.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
        """
        super().__init__()
        self.config = EfficientFormerConfig(
            depths=depths,
            hidden_sizes=hidden_sizes,
            downsamples=downsamples,
            dim=dim,
            key_dim=key_dim,
            attention_ratio=attention_ratio,
            resolution=resolution,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_expansion_ratio=mlp_expansion_ratio,
            hidden_dropout_prob=hidden_dropout_prob,
            patch_size=patch_size,
            num_channels=num_channels,
            pool_size=pool_size,
            downsample_patch_size=downsample_patch_size,
            downsample_stride=downsample_stride,
            downsample_pad=downsample_pad,
            drop_path_rate=drop_path_rate,
            num_meta3d_blocks=num_meta3d_blocks,
            distillation=distillation,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs,
        )
        model = EfficientFormerModel(self.config)
        self.model = self._model_clip(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, all_hidden_state = self.model(x, output_hidden_states=True, return_dict=False)
        all_hidden_state = [all_hidden_state[i] for i in [1, 3, 5, 6]]
        return all_hidden_state

    @staticmethod
    def list_models(author='snap-research', search='efficientformer') -> List[str]:
        return list_models_transformers(author=author, search=search)

    @classmethod
    def from_pretrained(cls, name, **kwargs) -> 'EfficientFormer':
        model = cls(**kwargs)
        _model = EfficientFormerModel.from_pretrained(name, **kwargs)
        model.model = cls._model_clip(_model)
        return model

    @staticmethod
    def _model_clip(m) -> None:

        class _Identity(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, **kwargs):
                return x

        m.flat = nn.Identity()
        m.meta3D_layers = nn.Identity()
        m.layernorm = nn.Identity()
        m.encoder.last_stage = _Identity()
        return m
