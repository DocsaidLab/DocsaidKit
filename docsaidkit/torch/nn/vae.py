from typing import Tuple

import torch
import torch.nn as nn

from .components import GAP
from .utils import PowerModule

__all__ = ['VAE']


class VAE(PowerModule):

    def __init__(self, in_channels: int, out_channels: int, do_pooling: bool = False):
        """
        Variational Autoencoder Module

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels, which corresponds to the size of the latent space.
            do_pooling (bool, optional): Whether to apply global average pooling. Defaults to False.
        """
        super().__init__()
        self.pool = GAP() if do_pooling else nn.Identity()
        self.encoder_mu = nn.Linear(in_channels, out_channels, bias=False)
        self.encoder_var = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

            Returns:
                feat (torch.Tensor): Encoded feature tensor of shape (batch_size, out_channels)
                kld_loss (torch.Tensor): KL divergence loss tensor of shape (1,)
        """
        x = self.pool(x)

        # Compute mean and variance of the encoded features using separate linear layers
        log_var = self.encoder_var(x)
        mu = self.encoder_mu(x)

        # Compute the standard deviation from the variance
        std = torch.exp(0.5 * log_var)

        # Sample random noise from a standard normal distribution
        eps = torch.randn_like(std)

        # Compute the encoded feature vector by adding the noise scaled by the standard deviation
        feat = mu + eps * std

        # Compute KL divergence loss between the learned distribution and a standard normal distribution
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))

        return feat, kld_loss
