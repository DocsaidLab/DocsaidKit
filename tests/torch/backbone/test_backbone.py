import pytest
import torch
from docsaidkit.torch import build_backbone, list_backbones

INPUT1 = torch.rand(1, 3, 320, 320)
INPUT2 = torch.rand(1, 6, 224, 224)
data = [
    # gpunet
    (
        INPUT1,
        {'name': 'gpunet_0', },
        {
            'out_shapes': [
                torch.Size([1, 32, 160, 160]),
                torch.Size([1, 32, 80, 80]),
                torch.Size([1, 64, 40, 40]),
                torch.Size([1, 256, 20, 20]),
                torch.Size([1, 704, 10, 10]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_1', 'out_indices': [0, 2, 3]},
        {
            'out_shapes': [
                torch.Size([1, 24, 160, 160]),
                torch.Size([1, 96, 40, 40]),
                torch.Size([1, 288, 20, 20]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_2', 'out_indices': [0, 1, 2]},
        {
            'out_shapes': [
                torch.Size([1, 32, 160, 160]),
                torch.Size([1, 32, 80, 80]),
                torch.Size([1, 112, 40, 40]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_p0'},
        {
            'out_shapes': [
                torch.Size([1, 32, 160, 160]),
                torch.Size([1, 64, 80, 80]),
                torch.Size([1, 96, 40, 40]),
                torch.Size([1, 256, 20, 20]),
                torch.Size([1, 704, 10, 10]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_p1'},
        {
            'out_shapes': [
                torch.Size([1, 32, 160, 160]),
                torch.Size([1, 64, 80, 80]),
                torch.Size([1, 96, 40, 40]),
                torch.Size([1, 256, 20, 20]),
                torch.Size([1, 704, 10, 10]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_d1'},
        {
            'out_shapes': [
                torch.Size([1, 33, 160, 160]),
                torch.Size([1, 44, 80, 80]),
                torch.Size([1, 67, 40, 40]),
                torch.Size([1, 190, 20, 20]),
                torch.Size([1, 268, 10, 10]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_d2', 'out_indices': [3, 4]},
        {
            'out_shapes': [
                torch.Size([1, 272, 20, 20]),
                torch.Size([1, 384, 10, 10]),
            ]
        }
    ),
    (
        INPUT1,
        {'name': 'gpunet_d2', 'out_indices': [-1]},
        {
            'out_shapes': [torch.Size([1, 384, 10, 10])]
        }
    ),
]


@ pytest.mark.parametrize('in_tensor,build_kwargs,expected', data)
def test_build_backbone(in_tensor, build_kwargs, expected):
    model = build_backbone(**build_kwargs)
    outs = model(in_tensor)
    if isinstance(outs, (list, tuple)):
        out_shapes = [x.shape for x in outs]
    else:
        out_shapes = outs.shape
    assert out_shapes == expected['out_shapes']


data = [
    (
        '*gpunet*',
        ['gpunet_0', 'gpunet_1', 'gpunet_2', 'gpunet_p0',
            'gpunet_p1', 'gpunet_d1', 'gpunet_d2']
    ),
]


@ pytest.mark.parametrize('filter,expected', data)
def test_list_backbones(filter, expected):
    assert list_backbones(filter) == expected
