from torch.optim import (ASGD, LBFGS, SGD, Adadelta, Adagrad, Adam, Adamax,
                         AdamW, RMSprop, Rprop, SparseAdam)
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LambdaLR, MultiStepLR,
                                      OneCycleLR, ReduceLROnPlateau, StepLR)

from .polynomial_lr_warmup import PolynomialLRWarmup
from .warm_up import *


def build_optimizer(model_params, name, **optim_options):
    cls_ = globals().get(name, None)
    if cls_ is None:
        raise ValueError(f'{name} is not supported optimizer.')
    return cls_(model_params, **optim_options)


def build_lr_scheduler(optimizer, name, **lr_scheduler_options):
    cls_ = globals().get(name, None)
    if cls_ is None:
        raise ValueError(f'{name} is not supported lr scheduler.')
    return cls_(optimizer, **lr_scheduler_options)
