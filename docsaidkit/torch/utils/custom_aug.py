import math
import random
from typing import Tuple

import albumentations as A
import numpy as np
from PIL import Image

from ...vision.geometric import imresize
from .mixin import BorderValueMixin, FillValueMixin

__all__ = [
    'RandomSunFlare', 'CoarseDropout', 'ShiftScaleRotate', 'SaftRotate',
    'Perspective', 'Shear', 'Rotate180',
]


class RandomSunFlare(A.RandomSunFlare):

    @property
    def src_radius(self):
        return random.randint(50, 200)

    @src_radius.setter
    def src_radius(self, x):
        return None


class CoarseDropout(FillValueMixin, A.CoarseDropout):
    ...


class ShiftScaleRotate(BorderValueMixin, A.ShiftScaleRotate):
    ...


class SaftRotate(BorderValueMixin, A.SafeRotate):
    ...


class Perspective(BorderValueMixin, A.Perspective):
    ...


class Shear:

    def __init__(self, max_shear: Tuple[int, int] = (20, 20), p: float = 0.5):
        self.p = p
        self.max_shear_left, self.max_shear_right = max_shear

    def __call__(self, img):
        if np.random.rand() < self.p:
            height, width, *_ = img.shape
            img = Image.fromarray(img)

            angle_to_shear = int(
                np.random.uniform(-self.max_shear_left - 1, self.max_shear_right + 1))
            if angle_to_shear != -1:
                angle_to_shear += 1

            phi = math.tan(math.radians(angle_to_shear))
            shift_in_pixels = phi * height
            shift_in_pixels = math.ceil(shift_in_pixels) \
                if shift_in_pixels > 0 else math.floor(shift_in_pixels)

            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)
            img = img.transform((int(round(width + shift_in_pixels)), height),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

            img = img.crop((abs(shift_in_pixels), 0, width, height))
            img = imresize(np.array(img), size=(height, width))

        return img


class Rotate180:

    def __init__(self, p: float = 0.5):
        self.p = p
        self.rotate180 = A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ], p=1)

    def __call__(self, **kwargs):
        is_rotate = 0
        if np.random.rand() < self.p:
            results = self.rotate180(**kwargs)
            is_rotate = 1
        results.update({'is_rotate': is_rotate})
        return results
