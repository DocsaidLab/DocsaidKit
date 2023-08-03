import torch

from docsaidkit.torch.neck import BiFPN, BiFPNs


def test_bifpn():
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    bifpn = BiFPN(in_channels_list, out_channels, extra_layers=2, out_indices=[0, 1, 2, 3])

    x1 = torch.randn(3, in_channels_list[0], 128, 128)
    x2 = torch.randn(3, in_channels_list[1], 64, 64)
    x3 = torch.randn(3, in_channels_list[2], 32, 32)
    x4 = torch.randn(3, in_channels_list[3], 16, 16)
    feats = [x1, x2, x3, x4]
    outs = bifpn(feats)
    assert len(outs) == 4
    assert bifpn.conv1x1s[0].__class__.__name__ == 'Identity'
    for out in outs:
        assert out.shape[0] == 3
        assert out.shape[1] == out_channels
        assert out.shape[2] == out.shape[3]


def test_build_bifpn():
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    extra_layers = 2
    upsample_mode = 'bilinear'
    out_indices = [0, 1, 2, 3]
    bifpn = BiFPN.build_bifpn(in_channels_list, out_channels, extra_layers, out_indices, upsample_mode)
    assert isinstance(bifpn, BiFPN)


def test_build_convbifpn():
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    extra_layers = 2
    upsample_mode = 'bilinear'
    out_indices = [0, 1, 2, 3]
    bifpn = BiFPN.build_convbifpn(in_channels_list, out_channels, extra_layers, out_indices, upsample_mode)
    assert isinstance(bifpn, BiFPN)


def test_bifpns_module():
    # Define test inputs
    in_channels_list = [64, 128, 256]
    out_channels = 256
    n_bifpn = 3
    extra_layers = 2
    out_indices = [0, 2]
    upsample_mode = 'nearest'
    attention = True

    # Initialize BiFPNs module
    bifpns = BiFPNs(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        n_bifpn=n_bifpn,
        extra_layers=extra_layers,
        out_indices=out_indices,
        upsample_mode=upsample_mode,
        attention=attention,
    )

    # Generate test inputs
    input_shapes = [(1, in_channels, 32 // 2**i, 32 // 2**i) for i, in_channels in enumerate(in_channels_list)]
    inputs = [torch.randn(shape) for shape in input_shapes]

    # Test forward pass
    output_shapes = [(1, out_channels, 32 // 2**i, 32 // 2**i) for i in range(len(in_channels_list))]
    expected_output_shapes = [shape for i, shape in enumerate(output_shapes) if i in out_indices]
    expected_output = [torch.randn(shape) for shape in expected_output_shapes]
    output = bifpns(inputs)
    assert isinstance(output, list)
    assert len(output) == 2
    for i, idx in enumerate(out_indices):
        assert output[i].shape == output_shapes[idx]

    # Test if the upsample_mode is correct
    for i in range(n_bifpn):
        assert bifpns.block[i].upsample_mode == upsample_mode

    # Test if the attention mechanism is applied
    for i in range(n_bifpn):
        assert bifpns.block[i].attention == attention

    # Test if the input channels are correct
    for i in range(n_bifpn):
        expected_in_channels = [out_channels] * (len(in_channels_list) + extra_layers) if i != 0 else in_channels_list
        assert bifpns.block[i].in_channels_list == expected_in_channels
