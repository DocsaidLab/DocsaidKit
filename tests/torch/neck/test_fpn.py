import torch

from docsaidkit.torch.neck import FPN, FPNs


def test_fpn():
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    fpn = FPN(in_channels_list, out_channels, extra_layers=2, out_indices=[0, 1, 2, 3])
    x1 = torch.randn(1, in_channels_list[0], 128, 128)
    x2 = torch.randn(1, in_channels_list[1], 64, 64)
    x3 = torch.randn(1, in_channels_list[2], 32, 32)
    x4 = torch.randn(1, in_channels_list[3], 16, 16)
    feats = [x1, x2, x3, x4]
    outs = fpn(feats)
    assert len(outs) == 4
    assert fpn.conv1x1s[0].__class__.__name__ == 'Identity'
    for out in outs:
        assert out.shape[0] == 1
        assert out.shape[1] == out_channels
        assert out.shape[2] == out.shape[3]


def test_build_fpn():
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    extra_layers = 2
    upsample_mode = 'bilinear'
    out_indices = [0, 1, 2, 3]
    fpn = FPN.build_fpn(in_channels_list, out_channels, extra_layers, out_indices, upsample_mode)
    assert isinstance(fpn, FPN)


def test_build_dwfpn():
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    extra_layers = 2
    upsample_mode = 'bilinear'
    out_indices = [0, 1, 2, 3]
    fpn = FPN.build_dwfpn(in_channels_list, out_channels, extra_layers, out_indices, upsample_mode)
    assert isinstance(fpn, FPN)


def test_fpns_module():
    # Define test inputs
    in_channels_list = [64, 128, 256]
    out_channels = 256
    n_fpn = 3
    extra_layers = 2
    out_indices = [0, 2]
    upsample_mode = 'nearest'

    # Initialize FPNs module
    fpns = FPNs(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        n_fpn=n_fpn,
        extra_layers=extra_layers,
        out_indices=out_indices,
        upsample_mode=upsample_mode,
    )

    # Generate test inputs
    input_shapes = [(1, in_channels, 32 // 2**i, 32 // 2**i) for i, in_channels in enumerate(in_channels_list)]
    inputs = [torch.randn(shape) for shape in input_shapes]

    # Test forward pass
    output_shapes = [(1, out_channels, 32 // 2**i, 32 // 2**i) for i in range(len(in_channels_list))]
    output = fpns(inputs)
    assert isinstance(output, list)
    assert len(output) == 2
    for i, idx in enumerate(out_indices):
        assert output[i].shape == output_shapes[idx]

    # Test if the upsample_mode is correct
    for i in range(n_fpn):
        assert fpns.block[i].upsample_mode == upsample_mode

    # Test if the input channels are correct
    for i in range(n_fpn):
        expected_in_channels = [out_channels] * (len(in_channels_list) + extra_layers) if i != 0 else in_channels_list
        assert fpns.block[i].in_channels_list == expected_in_channels
