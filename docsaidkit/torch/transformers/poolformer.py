from typing import List

import torch
import torch.nn as nn
from transformers import PoolFormerConfig, PoolFormerModel

from .utils import list_models_transformers

__all__ = ["PoolFormer"]


class PoolFormer(nn.Module):

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 16,
        stride: int = 16,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        depths: List[int] = [2, 2, 6, 2],
        hidden_sizes: List[int] = [64, 128, 320, 512],
        patch_sizes: List[int] = [7, 3, 3, 3],
        strides: List[int] = [4, 2, 2, 2],
        padding: List[int] = [2, 1, 1, 1],
        num_encoder_blocks: int = 4,
        drop_path_rate: float = 0.0,
        hidden_act: str = 'relu',
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        initializer_range: float = 0.02,
        **kwargs: dict,
    ) -> None:
        """
        PoolFormer is a model that replaces attention token mixer in transfomrers
        with extremely simple operator, pooling.

        Transformers have shown great potential in computer vision tasks. A common
        belief is their attention-based token mixer module contributes most to
        their competence. However, recent works show the attention-based module
        in transformers can be replaced by spatial MLPs and the resulted models
        still perform quite well. Based on this observation, we hypothesize that
        the general architecture of the transformers, instead of the specific
        token mixer module, is more essential to the model's performance.

        To verify this, we deliberately replace the attention module in transformers
        with an embarrassingly simple spatial pooling operator to conduct only
        the most basic token mixing. Surprisingly, we observe that the derived
        model, termed as PoolFormer, achieves competitive performance on multiple
        computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves
        82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like
        baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer
        parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer
        verifies our hypothesis and urges us to initiate the concept of "MetaFormer",
        a general architecture abstracted from transformers without specifying
        the token mixer. Based on the extensive experiments, we argue that
        MetaFormer is the key player in achieving superior results for recent
        transformer and MLP-like models on vision tasks.

        This work calls for more future research dedicated to improving MetaFormer
        instead of focusing on the token mixer modules. Additionally, our proposed
        PoolFormer could serve as a starting baseline for future MetaFormer
        architecture design.

        Args:
            num_channels (int, optional):
                The number of channels in the input data. Defaults to 3.
            patch_size (int, optional):
                The size of the patches extracted from the input data.
                Defaults to 16.
            stride (int, optional):
                The stride of the convolutional layer used to extract patches
                from the input data. Defaults to 16.
            pool_size (int, optional):
                The size of the pooling kernel used in the PoolFormer encoder
                layers. Defaults to 3.
            mlp_ratio (float, optional):
                The ratio of the hidden size in the feedforward layer of the
                PoolFormer encoder to the input size. Defaults to 4.0.
            depths (List[int], optional):
                The number of blocks in each stage of the PoolFormer encoder.
                Defaults to [2, 2, 6, 2].
            hidden_sizes (List[int], optional):
                The size of the hidden layer in each block of the PoolFormer
                encoder. Defaults to [64, 128, 320, 512].
            patch_sizes (List[int], optional):
                The size of the convolutional kernel in each block of the
                PoolFormer encoder. Defaults to [7, 3, 3, 3].
            strides (List[int], optional):
                The stride of the convolutional layer in each block of the
                PoolFormer encoder. Defaults to [4, 2, 2, 2].
            padding (List[int], optional):
                The padding size of the convolutional layer in each block of the
                PoolFormer encoder. Defaults to [2, 1, 1, 1].
            num_encoder_blocks (int, optional):
                The number of encoder blocks in the PoolFormer encoder.
                Defaults to 4.
            drop_path_rate (float, optional):
                The drop path rate used in the PoolFormer encoder.
                Defaults to 0.0.
            hidden_act (str, optional):
                The activation function used in the PoolFormer encoder.
                Defaults to "relu".
            use_layer_scale (bool, optional):
                Whether to use layer scaling in the PoolFormer encoder.
                Defaults to True.
            layer_scale_init_value (float, optional):
                The initial value of the layer scale in the PoolFormer encoder.
                Defaults to 1e-5.
            initializer_range (float, optional):
                The range of the uniform distribution used to initialize the
                weights in the PoolFormer encoder. Defaults to 0.02.
        """
        super().__init__()
        self.config = PoolFormerConfig(
            num_channels=num_channels,
            patch_size=patch_size,
            stride=stride,
            pool_size=pool_size,
            mlp_ratio=mlp_ratio,
            depths=depths,
            hidden_sizes=hidden_sizes,
            patch_sizes=patch_sizes,
            strides=strides,
            padding=padding,
            num_encoder_blocks=num_encoder_blocks,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            initializer_range=initializer_range,
            **kwargs,
        )
        self.model = PoolFormerModel(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, all_hidden_state = self.model(x, output_hidden_states=True, return_dict=False)
        return all_hidden_state

    @staticmethod
    def list_models(author='sail', search='poolformer') -> List[str]:
        return list_models_transformers(author=author, search=search)

    @classmethod
    def from_pretrained(cls, name, **kwargs) -> 'PoolFormer':
        model = cls(**kwargs)
        model.model = PoolFormerModel.from_pretrained(name, **kwargs)
        return model
