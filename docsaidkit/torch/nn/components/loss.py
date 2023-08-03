from typing import Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import (BCELoss, BCEWithLogitsLoss,
                                   CrossEntropyLoss, CTCLoss, KLDivLoss,
                                   L1Loss, MSELoss, SmoothL1Loss)

__all__ = [
    'build_loss', 'AWingLoss', 'WeightedAWingLoss',
    'BCELoss', 'BCEWithLogitsLoss', 'CrossEntropyLoss',
    'CTCLoss', 'KLDivLoss', 'L1Loss', 'MSELoss', 'SmoothL1Loss',
]

class AWingLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 2.1,
        omega: float = 14,
        epsilon: float = 1,
        theta: float = 0.5
    ):
        """
        Initialize the parameters of the AWingLoss loss function.

        Args:
            alpha (float, optional):
                The alpha parameter. Defaults to 2.1.
            omega (float, optional):
                The omega parameter. Defaults to 14.
            epsilon (float, optional):
                The epsilon parameter. Defaults to 1.
            theta (float, optional):
                The theta parameter. Defaults to 0.5.
        """
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, preds, targets):
        diff = torch.abs(targets - preds)
        case1_mask = diff < self.theta
        case2_mask = ~case1_mask
        loss_case1 = self.omega * torch.log1p((diff[case1_mask] / self.epsilon) ** self.alpha)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon)**(self.alpha - targets))) \
            * (self.alpha - targets) * ((self.theta / self.epsilon)**(self.alpha - targets - 1)) \
            * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log1p((self.theta / self.epsilon)**(self.alpha - targets))
        loss_case2 = A[case2_mask] * diff[case2_mask] - C[case2_mask]
        loss_matrix = torch.zeros_like(preds)
        loss_matrix[case1_mask] = loss_case1
        loss_matrix[case2_mask] = loss_case2
        return loss_matrix


class WeightedAWingLoss(nn.Module):

    def __init__(
        self,
        w: float = 10,
        alpha: float = 2.1,
        omega: float = 14,
        epsilon: float = 1,
        theta: float = 0.5
    ):
        super().__init__()
        self.w = w
        self.AWingLoss = AWingLoss(alpha, omega, epsilon, theta)

    def forward(self, preds, targets, weight_map=None):
        loss = self.AWingLoss(preds, targets)
        if weight_map is None:
            weight_map = targets > 0
        weighted = loss * (self.w * weight_map.to(loss.dtype) + 1)
        return weighted.mean()


def build_loss(name: str, **options) -> Union[nn.Module, None]:
    """Build a loss func layer given the name and options."""
    cls = globals().get(name, None)
    if cls is None:
        raise KeyError(f'Unsupported loss func: {name}')
    return cls(**options)
