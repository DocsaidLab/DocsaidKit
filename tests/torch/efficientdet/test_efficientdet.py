import pytest
import torch

from docsaidkit.torch import EfficientDet


@pytest.fixture
def input_tensor():
    # create a sample input tensor
    return torch.rand((1, 3, 512, 512))


@pytest.mark.parametrize("compound_coef, pretrained", [
    (0, True),
    (1, True),
    (2, True),
    (3, True),
    (4, True),
    (5, True),
    (6, False),
    (7, False),
    (8, False),
    (0, False),
])
def test_efficientdet_backbone(input_tensor, compound_coef, pretrained):
    # create the model with the specified compound_coef and pretrained options
    model = EfficientDet(compound_coef=compound_coef, pretrained=pretrained)

    # verify that the model is PowerModule and nn.Module
    assert isinstance(model, EfficientDet)
    assert isinstance(model, torch.nn.Module)

    # verify that the forward pass of the model returns a list of feature maps
    output = model(input_tensor)
    assert isinstance(output, list)

    # verify that the shape of each feature map in the output list is correct
    conv_channel_coef = {
        0: [40, 112, 320],
        1: [40, 112, 320],
        2: [48, 120, 352],
        3: [48, 136, 384],
        4: [56, 160, 448],
        5: [64, 176, 512],
        6: [72, 200, 576],
        7: [80, 224, 640],
        8: [88, 248, 704],
    }

    for i in range(len(output)):
        expected_shape = (
            1,
            model.fpn_num_filters[compound_coef],
            int(input_tensor.shape[2] / 2 ** (i+3)),
            int(input_tensor.shape[3] / 2 ** (i+3))
        )

        assert output[i].shape == expected_shape
