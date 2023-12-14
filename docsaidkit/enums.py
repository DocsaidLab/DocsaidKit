from enum import Enum, IntEnum

import cv2

from .mixins import EnumCheckMixin

__all__ = [
    'INTER', 'ROTATE', 'BORDER', 'MORPH', 'COLORSTR', 'FORMATSTR', 'IMGTYP'
]


class INTER(EnumCheckMixin, IntEnum):
    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4


class ROTATE(EnumCheckMixin, IntEnum):
    ROTATE_90 = cv2.ROTATE_90_CLOCKWISE
    ROTATE_180 = cv2.ROTATE_180
    ROTATE_270 = cv2.ROTATE_90_COUNTERCLOCKWISE


class BORDER(EnumCheckMixin, IntEnum):
    DEFAULT = cv2.BORDER_DEFAULT
    CONSTANT = cv2.BORDER_CONSTANT
    REFLECT = cv2.BORDER_REFLECT
    REFLECT_101 = cv2.BORDER_REFLECT_101
    REPLICATE = cv2.BORDER_REPLICATE
    WRAP = cv2.BORDER_WRAP


class MORPH(EnumCheckMixin, IntEnum):
    CROSS = cv2.MORPH_CROSS
    RECT = cv2.MORPH_RECT
    ELLIPSE = cv2.MORPH_ELLIPSE


class COLORSTR(EnumCheckMixin, Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    BRIGHT_BLACK = 90
    BRIGHT_RED = 91
    BRIGHT_GREEN = 92
    BRIGHT_YELLOW = 93
    BRIGHT_BLUE = 94
    BRIGHT_MAGENTA = 95
    BRIGHT_CYAN = 96
    BRIGHT_WHITE = 97


class FORMATSTR(EnumCheckMixin, Enum):
    BOLD = 1
    ITALIC = 3
    UNDERLINE = 4


class IMGTYP(EnumCheckMixin, IntEnum):
    JPEG = 0
    PNG = 1
