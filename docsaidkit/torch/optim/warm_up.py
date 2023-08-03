from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler

__all__ = ['WrappedLRScheduler', 'MultiStepLRWarmUp']


class WrappedLRScheduler(_LRScheduler):
    """
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestone (int):
            milestone step for warm-up.
        multiplier (float):
            A factor to multiply base_lr.
            if multiplier > 1.0, learning rate = base lr * multiplier.
            if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        after_scheduler (lr_scheduler):
            after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestone: int,
        multiplier: float = 1.0,
        after_scheduler: _LRScheduler = None,
        interval='step'
    ):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.milestone = milestone
        self.after_scheduler = after_scheduler
        self.finished = False
        self.interval = interval
        super().__init__(optimizer)  # need be set in the end of __init__

    def get_lr(self):
        # do after_scheduler
        if self.last_epoch > self.milestone:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.milestone) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.milestone + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.milestone)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step()


def MultiStepLRWarmUp(
    optimizer: Optimizer,
    milestones: List[int],
    warmup_milestone: int,
    gamma: float = 0.1,
    last_epoch: int = -1,
    interval='step',
    verbose: bool = False,
):
    scheduler = MultiStepLR(optimizer, milestones, gamma, last_epoch, verbose)
    return WrappedLRScheduler(optimizer,
                              warmup_milestone,
                              after_scheduler=scheduler,
                              interval=interval)
