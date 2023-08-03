import math

import torch

__all__ = ['sinusoidal_positional_encoding_1d']


def sinusoidal_positional_encoding_1d(length, dim):
    """ Sinusoidal positional encoding for non-recurrent neural networks.
        REFERENCES: Attention Is All You Need
        URL: https://arxiv.org/abs/1706.03762
    """
    if dim % 2 != 0:
        raise ValueError(
            'Cannot use sin/cos positional encoding with '
            f'odd dim (got dim={dim})')

    # position embedding
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
