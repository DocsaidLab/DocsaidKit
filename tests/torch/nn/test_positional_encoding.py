import pytest
import torch

from docsaidkit.torch.nn import sinusoidal_positional_encoding_1d


@pytest.mark.parametrize("length, dim", [(10, 16), (20, 32), (5, 8)])
def test_sinusoidal_positional_encoding_1d(length, dim):
    # Test that the output has the correct shape
    pe = sinusoidal_positional_encoding_1d(length, dim)
    assert pe.shape == (length, dim)

    # Test that the output has the correct values
    for i in range(length):
        for j in range(dim // 2):
            sin_val = torch.sin(torch.tensor(i / (10000 ** (2 * j / dim))))
            cos_val = torch.cos(torch.tensor(i / (10000 ** (2 * j / dim))))
            assert torch.isclose(pe[i][2*j], sin_val, atol=1e-6)
            assert torch.isclose(pe[i][2*j+1], cos_val, atol=1e-6)

    # Test that odd dimensions raise a ValueError
    with pytest.raises(ValueError):
        pe = sinusoidal_positional_encoding_1d(length, dim=7)
