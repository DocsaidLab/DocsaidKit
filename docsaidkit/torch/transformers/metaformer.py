from typing import List, Optional, Union
from warnings import warn

import torch
import torch.nn as nn

from ..nn import CNN2Dcell, LayerNorm2d, PowerModule, StarReLU
from .token_mixer import (AttentionMixing, PoolMixing, RandomMixing,
                          SepConvMixing)

__all__ = ['MetaFormer', 'MetaFormerBlock', 'MlpBlock']


MODEL_SETTINGS = {
    'poolformer_v2_tiny': {
        'depths': [1, 1, 3, 1],
        'hidden_sizes': [16, 32, 96, 128],
        'token_mixers': 'PoolMixing',
        'mlp_forwards': {'name': 'MlpBlock', 'expand_ratio': 1.5}
    },
    'poolformer_v2_small': {
        'depths': [2, 2, 4, 2],
        'hidden_sizes': [32, 64, 128, 256],
        'token_mixers': 'PoolMixing',
        'mlp_forwards': {'name': 'MlpBlock', 'expand_ratio': 1.5}
    },
    'poolformer_v2_s12': {
        'depths': [2, 2, 6, 2],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': 'PoolMixing',
    },
    'poolformer_v2_s24': {
        'depths': [4, 4, 12, 4],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': 'PoolMixing',
    },
    'poolformer_v2_s36': {
        'depths': [6, 6, 18, 6],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': 'PoolMixing',
    },
    'poolformer_v2_m36': {
        'depths': [6, 6, 18, 6],
        'hidden_sizes': [96, 192, 384, 768],
        'token_mixers': 'PoolMixing',
    },
    'poolformer_v2_m48': {
        'depths': [8, 8, 24, 8],
        'hidden_sizes': [96, 192, 384, 768],
        'token_mixers': 'PoolMixing',
    },
    'convformer_s18': {
        'depths': [3, 3, 9, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': 'SepConvMixing',
    },
    'convformer_s36': {
        'depths': [3, 12, 18, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': 'SepConvMixing',
    },
    'convformer_m36': {
        'depths': [3, 12, 18, 3],
        'hidden_sizes': [96, 192, 384, 576],
        'token_mixers': 'SepConvMixing',
    },
    'convformer_b36': {
        'depths': [3, 12, 18, 3],
        'hidden_sizes': [128, 256, 512, 768],
        'token_mixers': 'SepConvMixing',
    },
    'caformer_tiny': {
        'depths': [1, 1, 2, 1],
        'hidden_sizes': [16, 32, 64, 128],
        'token_mixers': ['SepConvMixing', 'SepConvMixing', 'AttentionMixing', 'AttentionMixing'],
        'mlp_forwards': {'name': 'MlpBlock', 'expand_ratio': 1.5}
    },
    'caformer_small': {
        'depths': [1, 1, 4, 2],
        'hidden_sizes': [16, 48, 128, 160],
        'token_mixers': ['SepConvMixing', 'SepConvMixing', 'AttentionMixing', 'AttentionMixing'],
        'mlp_forwards': {'name': 'MlpBlock', 'expand_ratio': 1.5}
    },
    'caformer_s18': {
        'depths': [3, 3, 9, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': ['SepConvMixing', 'SepConvMixing', 'AttentionMixing', 'AttentionMixing'],
    },
    'caformer_s36': {
        'depths': [3, 12, 18, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'token_mixers': ['SepConvMixing', 'SepConvMixing', 'AttentionMixing', 'AttentionMixing'],
    },
    'caformer_m36': {
        'depths': [3, 12, 18, 3],
        'hidden_sizes': [96, 192, 384, 576],
        'token_mixers': ['SepConvMixing', 'SepConvMixing', 'AttentionMixing', 'AttentionMixing'],
    },
    'caformer_b36': {
        'depths': [3, 12, 18, 3],
        'hidden_sizes': [128, 256, 512, 768],
        'token_mixers': ['SepConvMixing', 'SepConvMixing', 'AttentionMixing', 'AttentionMixing'],
    },
}


def build_token_mixer(name, **options) -> Union[nn.Module, None]:
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'Token mixer named {name} is not supported.')
    return cls(**options)


def build_mlps_forward(name, **options) -> Union[nn.Module, None]:
    cls = globals().get(name, None)
    if cls is None:
        raise ValueError(f'MLP forward named {name} is not supported.')
    return cls(**options)


class MlpBlock(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            expand_ratio: float = 4
        ) -> None:
        """
        MLP as used in MetaFormer models baslines and related networks.

        Args:
            in_features:
                The number of input features.
            out_features:
                The number of output features.
            expand_ratio:
                The multiplier applied to the number of input features to obtain
                the number of hidden features. Defaults to 4.
        """
        super().__init__()
        hidden_features = int(expand_ratio * in_features)
        self.fc1_block = nn.Conv2d(in_features, hidden_features, 1)
        self.fc2_block = nn.Conv2d(hidden_features, out_features, 1)
        self.act = StarReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1_block(x)
        x = self.act(x)
        x = self.fc2_block(x)
        return x


class MetaFormerBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        token_mixer: Union[str, dict] = None,
        mlp_forward: Union[str, dict] = None,
    ) -> None:
        """
        A single block of the MetaFormer model, consisting of a weighted sum of
        a token mixing module and an MLP.

        Args:
            in_features (int):
                The number of input features.
            token_mixer (Union[dict, nn.Module], optional):
                The token mixing module to use in the block. Can be either an
                nn.Module instance or a dictionary specifying the token mixing
                module to build using the `build_token_mixer` function.
                Defaults to None.
            mlp_forward (Union[dict, nn.Module], optional):
                The MLP module to use in the block. Can be either an nn.Module
                instance or a dictionary specifying the MLP module to build using
                the `build_mlps_forward` function.
                Defaults to None.
        """

        super().__init__()
        self.in_features = in_features
        self.token_mixer = self._build_token_mixers(token_mixer)
        self.mlp_forward = self._build_mlp_forwars(mlp_forward)
        self.norm_mixer = LayerNorm2d(in_features)
        self.norm_mlp = LayerNorm2d(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm_mixer(x))
        x = x + self.mlp_forward(self.norm_mlp(x))
        return x

    def _build_mlp_forwars(self, param: Union[str, dict]) -> nn.Module:
        if param is None:
            return nn.Identity()
        if isinstance(param, str):
            if param == 'Identity':
                return nn.Identity()
            elif param == 'MlpBlock':
                return MlpBlock(
                    in_features=self.in_features,
                    out_features=self.in_features,
                )
            else:
                raise ValueError(f'Unsupport mlp_forwards settings: {param}')
        elif isinstance(param, dict):
            if param['name'] in ['MlpBlock']:
                param.update({
                    'in_features': self.in_features,
                    'out_features': self.in_features
                })
            return build_mlps_forward(**param)

    def _build_token_mixers(self, param: Union[str, dict]) -> nn.Module:
        if param is None:
            return nn.Identity()
        if isinstance(param, str):
            if param == 'AttentionMixing':
                return AttentionMixing(self.in_features)
            elif param == 'SepConvMixing':
                return SepConvMixing(self.in_features)
            elif param == 'PoolMixing':
                return PoolMixing()
            elif param == 'RandomMixing':
                warn(
                    'Do not use RandomMixing in MetaFormer by pass string name,'
                    'to token_mixer, use `token_mixer={"name": "RandomMixing", "num_tokens": N}` instead.'
                    'Set token_mixer to nn.Identity() instead.'
                )
                return nn.Identity()
            elif param == 'Identity':
                return nn.Identity()
            else:
                raise ValueError(f'Unsupport token mixer settings: {param}')
        elif isinstance(param, dict):
            if param['name'] in ['AttentionMixing', 'SepConvMixing']:
                param.update({'in_features': self.in_features})
            return build_token_mixer(**param)

class MetaFormer(PowerModule):

    def __init__(
        self,
        num_channels: int = 3,
        depths: List[int] = [2, 2, 6, 2],
        hidden_sizes: List[int] = [64, 128, 320, 512],
        patch_sizes: List[int] = [7, 3, 3, 3],
        strides: List[int] = [4, 2, 2, 2],
        padding: List[int] = [2, 1, 1, 1],
        token_mixers: Union[dict, str, List[Union[dict, str]]] = 'PoolMixing',
        mlp_forwards: Union[dict, str, List[Union[dict, str]]] = 'MlpBlock',
        out_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the MetaFormer model.

        Args:
            num_channels (int, optional):
                The number of channels in the input image. Defaults to 3.
            depths (List[int], optional):
                The number of blocks in each stage of the MetaFormer.
                Defaults to [2, 2, 6, 2].
            hidden_sizes (List[int], optional):
                The number of channels in each stage of the MetaFormer.
                Defaults to [64, 128, 320, 512].
            patch_sizes (List[int], optional):
                The patch size used in each stage of the MetaFormer.
                Defaults to [7, 3, 3, 3].
            strides (List[int], optional):
                The stride used in each stage of the MetaFormer.
                Defaults to [4, 2, 2, 2].
            padding (List[int], optional):
                The padding used in each stage of the MetaFormer.
                Defaults to [2, 1, 1, 1].
            token_mixers (Union[dict, str, List[Union[dict, str]]], optional):
                The token mixing modules used in the model.
                Defaults to 'PoolMixing'.
            mlp_forwards (Union[dict, str, List[Union[dict, str]]], optional):
                The MLP modules used in the model.
                Defaults to 'MlpBlock'.
            out_indices (Optional[List[int]], optional):
                The indices of the output feature maps.
                Defaults to None.
        """
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            raise ValueError('depths must be either list or tuple.')

        if not isinstance(hidden_sizes, (list, tuple)):
            raise ValueError('hidden_sizes must be either list or tuple.')

        self.num_stage = len(depths)

        self.downsamples = nn.ModuleList([
            nn.Sequential(
                LayerNorm2d(hidden_sizes[i - 1]) if i > 0 else nn.Identity(),
                nn.Conv2d(
                    in_channels=num_channels if i == 0 else hidden_sizes[i-1],
                    out_channels=hidden_sizes[i],
                    kernel_size=ksize,
                    stride=s,
                    padding=p
                ),
                LayerNorm2d(hidden_sizes[i]) if i == 0 else nn.Identity(),
            ) for i, (ksize, s, p) in enumerate(zip(patch_sizes, strides, padding))
        ])

        token_mixers = [token_mixers] * self.num_stage \
            if not isinstance(token_mixers, (list, tuple)) else token_mixers
        mlp_forwards = [mlp_forwards] * self.num_stage \
            if not isinstance(mlp_forwards, (list, tuple)) else mlp_forwards

        self.stages = nn.ModuleList([
            nn.Sequential(*[
                MetaFormerBlock(
                    in_features=hidden_sizes[i],
                    token_mixer=token_mixers[i],
                    mlp_forward=mlp_forwards[i],
                )
                for _ in range(depth)
            ])
            for i, depth in enumerate(depths)
        ])

        self.out_indices = out_indices
        self.initialize_weights_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(self.num_stage):
            x = self.downsamples[i](x)
            x = self.stages[i](x)
            outs.append(x)

        if self.out_indices is not None:
            outs = [outs[i] for i in self.out_indices]

        return outs

    @classmethod
    def from_pretrained(cls, name: str, **kwargs) -> 'MetaFormer':
        """
        Initializes the MetaFormer model from the pretrained model.

        Args:
            model_name (str):
                The name of the pretrained model.
            **kwargs:
                The other arguments of the model.

        Returns:
            MetaFormer:
                The MetaFormer model.
        """
        if name not in MODEL_SETTINGS:
            raise ValueError(f'Unsupport model name: {name}')

        model_settings = MODEL_SETTINGS[name]
        model_settings.update(kwargs)
        return cls(**model_settings)
