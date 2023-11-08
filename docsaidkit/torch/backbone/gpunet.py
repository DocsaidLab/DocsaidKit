from typing import List, Optional

import torch
import torch.nn as nn

from ..nn import PowerModule
from ..tools import has_children

__all__ = ['GPUNet']


class GPUNet(PowerModule):

    MetaParams = {
        'gpunet_0': {
            'name': 'GPUNet-0',
            'stage_index': [3, 5, 8, 11, 13]
        },
        'gpunet_1': {
            'name': 'GPUNet-1',
            'stage_index': [2, 4, 6, 11, 15]
        },
        'gpunet_2': {
            'name': 'GPUNet-2',
            'stage_index': [4, 5, 8, 18, 33]
        },
        'gpunet_p0': {
            'name': 'GPUNet-P0',
            'stage_index': [3, 4, 7, 10, 14]
        },
        'gpunet_p1': {
            'name': 'GPUNet-P1',
            'stage_index': [3, 6, 8, 11, 15]
        },
        'gpunet_d1': {
            'name': 'GPUNet-D1',
            'stage_index': [2, 5, 9, 17, 23]
        },
        'gpunet_d2': {
            'name': 'GPUNet-D2',
            'stage_index': [2, 5, 9, 19, 26]
        },
    }

    def __init__(
        self,
        stages: List[torch.nn.Sequential],
        out_indices: Optional[List[int]] = None
    ):
        super().__init__()
        for i, stage in enumerate(stages):
            self.add_module(f'stage_{i}', stage)

        self.channels = []
        with torch.no_grad():
            self.out_indices = None
            for x in self.forward(torch.rand(1, 3, 224, 224)):
                self.channels.append(x.shape[1])

        self.out_indices = out_indices

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        outs = []
        for i in range(5):
            x = self._modules[f'stage_{i}'](x)
            outs.append(x)

        if self.out_indices is not None:
            outs = [outs[i] for i in self.out_indices]

        return outs

    def __repr__(self):
        return str(self._modules)

    @ classmethod
    def build_gpunet(
        cls,
        name: str,
        pretrained: bool = False,
        precision: str = 'fp32',
        out_indices: Optional[List[int]] = None,
    ):

        def _replace_padding(model):
            for m in model.children():
                if has_children(m):
                    _replace_padding(m)
                else:
                    if isinstance(m, nn.Conv2d) and getattr(m, 'stride') == (2, 2):
                        ksize = tuple(map(lambda x: int(x // 2),
                                      getattr(m, 'kernel_size')))
                        setattr(m, 'padding', ksize)

        allow_model_name = '\n\t'.join(
            [name for name in cls.MetaParams.keys()])
        if name not in cls.MetaParams:
            raise ValueError(
                f'Input `name`: {name} is invalid.\n'
                'Try the model names as follow: \n'
                f'\t{allow_model_name}'
                f'\n    Ref: https://pytorch.org/hub/nvidia_deeplearningexamples_gpunet/'
            )

        model = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_gpunet',
            pretrained=pretrained,
            model_type=cls.MetaParams[name]['name'],
            model_math=precision,
            trust_repo=True
        )

        stages = []
        start_idx = 0
        for stop_idx in cls.MetaParams[name]['stage_index']:
            layers = model.network[start_idx: stop_idx]
            _replace_padding(layers)
            stages.append(layers)
            start_idx = stop_idx

        return cls(
            stages=stages,
            out_indices=out_indices,
        )
