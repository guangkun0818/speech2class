# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" Optimizer and LR scheduler setup """

from typing import Union

import math
import torch
import warnings
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


def OptimSetup(config):
    # Optimizer and Scheduler Setup inferface
    if config["optimizer"]["type"] == "Adam":
        optimizer = Adam
    else:
        raise ValueError("{} optimizer is not supported.".format(
            config["optimizer"]["type"]))

    if config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau
    elif config["lr_scheduler"]["type"] == "Cosine_Annealing":
        lr_scheduler = CosineAnnealingLR
    elif config["lr_scheduler"]["type"] == "Warmup":
        lr_scheduler = WarmupLR
    else:
        raise ValueError("{} lr_scheduler is not supported.".format(
            config["lr_scheduler"]["type"]))
    return optimizer, lr_scheduler


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler
    This scheduler is almost same as NoamLR Scheduler except for following
    difference:
    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [lr * step_num**-0.5 for lr in self.base_lrs]
        else:
            return [
                lr * self.warmup_steps**0.5 *
                min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
                for lr in self.base_lrs
            ]

    def set_step(self, step: int):
        self.last_epoch = step
