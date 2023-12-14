import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, Mapping, Optional
from warnings import warn

import cv2
import numpy as np
from dacite import from_dict

from .structures import Box, Boxes, Polygon, Polygons

__all__ = [
    'EnumCheckMixin', 'INTER', 'ROTATE', 'BORDER', 'MORPH', 'COLORSTR',
    'FORMATSTR', 'IMGTYP'
]


class EnumCheckMixin:

    @classmethod
    def obj_to_enum(cls: Enum, obj: Any):
        if isinstance(obj, str):
            try:
                return getattr(cls, obj)
            except AttributeError:
                pass
        elif isinstance(obj, cls):
            return obj
        elif isinstance(obj, int):
            try:
                return cls(obj)
            except ValueError:
                pass

        raise ValueError(f"{obj} is not correct for {cls.__name__}")


def dict_to_jsonable(
    d: Mapping,
    jsonable_func: Optional[Dict[str, Callable]] = None,
    dict_factory: Mapping = OrderedDict,
) -> Any:
    out = dict_factory()
    for k, v in d.items():
        if jsonable_func is not None and k in jsonable_func:
            out[k] = jsonable_func[k](v)
        else:
            if isinstance(v, (Box, Boxes)):
                out[k] = v.convert('XYXY').numpy().astype(
                    float).round().tolist()
            elif isinstance(v, (Polygon, Polygons)):
                out[k] = v.numpy().astype(float).round().tolist()
            elif isinstance(v, (np.ndarray, np.generic)):
                out[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    dict_to_jsonable(x, jsonable_func) if isinstance(
                        x, dict) else x
                    for x in v
                ]
            elif isinstance(v, Enum):
                out[k] = v.name
            elif isinstance(v, Mapping):
                out[k] = dict_to_jsonable(v, jsonable_func)
            else:
                out[k] = v

    try:
        json.dumps(out)
    except Exception as e:
        warn(e)
    return out


class DataclassCopyMixin:

    def __copy__(self):
        out = asdict(self, dict_factory=OrderedDict)
        return from_dict(data_class=self.__class__, data=out)

    def __deepcopy__(self, memo):
        out = asdict(self, dict_factory=OrderedDict)
        return from_dict(data_class=self.__class__, data=out)


class DataclassToJsonMixin:

    def be_jsonable(self, dict_factory=OrderedDict):
        d = asdict(self, dict_factory=OrderedDict)
        return dict_to_jsonable(d, getattr(self, 'jsonable_func', None), dict_factory)

    def regist_jsonable_func(self, jsonable_func: Optional[Dict[str, Callable]] = None):
        self.jsonable_func = jsonable_func


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
