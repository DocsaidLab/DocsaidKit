import sys

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class CustomTQDMProgressBar(TQDMProgressBar):

    def __init__(self, unit_scale: float, *args, **kwargs):
        self.unit_scale = unit_scale
        super().__init__(*args, **kwargs)

    def create_tqdm(self, desc: str, leave: bool, position_offset: int = 0) -> Tqdm:
        position = 2 * self.process_position + position_offset
        return Tqdm(
            desc=desc,
            position=position,
            disable=self.is_disabled,
            leave=leave,
            dynamic_ncols=True,
            file=sys.stdout,
            unit_scale=self.unit_scale
        )

    def init_sanity_tqdm(self) -> Tqdm:
        return self.create_tqdm(self.sanity_check_description, leave=False)

    def init_train_tqdm(self) -> Tqdm:
        return self.create_tqdm(self.train_description, leave=True)

    def init_predict_tqdm(self) -> Tqdm:
        return self.create_tqdm(self.predict_description, leave=True)

    def init_validation_tqdm(self) -> Tqdm:
        has_main_bar = self.trainer.state.fn != "validate"
        return self.create_tqdm(self.validation_description, leave=not has_main_bar, position_offset=has_main_bar)

    def init_test_tqdm(self) -> Tqdm:
        return self.create_tqdm("Testing", leave=True)
