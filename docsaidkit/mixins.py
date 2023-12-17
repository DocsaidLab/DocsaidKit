import json
from collections import OrderedDict
from dataclasses import asdict
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional
from warnings import warn

import numpy as np
from dacite import from_dict

from .structures import Box, Boxes, Polygon, Polygons

__all__ = [
    'EnumCheckMixin', 'DataclassCopyMixin', 'DataclassToJsonMixin',
    'dict_to_jsonable',
]


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


class DataclassCopyMixin:

    def __copy__(self):
        return self.__class__(**{
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        })

    def __deepcopy__(self, memo):
        out = asdict(self, dict_factory=OrderedDict)
        return from_dict(data_class=self.__class__, data=out)


class DataclassToJsonMixin:

    def __init__(self):
        self.jsonable_func = None

    def be_jsonable(self, dict_factory=OrderedDict):
        d = asdict(self, dict_factory=dict_factory)
        return dict_to_jsonable(d, getattr(self, 'jsonable_func', None), dict_factory)

    def regist_jsonable_func(self, jsonable_func: Optional[Dict[str, Callable]] = None):
        self.jsonable_func = jsonable_func
