import random
from typing import Any, Dict, List

import cv2
import lightning as L

from ..optim import build_lr_scheduler, build_optimizer

__all__ = [
    'BaseMixin', 'BorderValueMixin', 'FillValueMixin',
]


class BaseMixin(L.LightningModule):

    def apply_solver_config(
        self,
        optimizer: Dict[str, Any],
        lr_scheduler: Dict[str, Any]
    ) -> None:
        self.optimizer_name, self.optimizer_opts = optimizer.values()
        self.sche_name, self.sche_opts, self.sche_pl_opts = lr_scheduler.values()

    def get_optimizer_params(self) -> List[Dict[str, Any]]:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimizer_opts.get('weight_decay', 0.0),
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        optimizer = build_optimizer(
            name=self.optimizer_name,
            model_params=self.get_optimizer_params(),  # 使用新的 get_optimizer_params 方法
            **self.optimizer_opts
        )
        scheduler = build_lr_scheduler(
            name=self.sche_name,
            optimizer=optimizer,
            **self.sche_opts
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self.sche_pl_opts
            }
        }

    def get_lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']


class BorderValueMixin:

    @property
    def pad_mode(self):
        return random.choice([
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
        ])

    @property
    def border_mode(self):
        return random.choice([
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
        ])

    @property
    def value(self):
        return [random.randint(0, 255) for _ in range(3)]

    @pad_mode.setter
    def pad_mode(self, x):
        return None

    @border_mode.setter
    def border_mode(self, x):
        return None

    @value.setter
    def value(self, x):
        return None


class FillValueMixin:

    @property
    def fill_value(self):
        return [random.randint(0, 255) for _ in range(3)]

    @fill_value.setter
    def fill_value(self, x):
        return None
