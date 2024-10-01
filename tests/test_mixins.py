from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, List

import numpy as np
import pytest

import docsaidkit as D
from docsaidkit import (DataclassCopyMixin, DataclassToJsonMixin,
                        EnumCheckMixin, dict_to_jsonable)

MockImage = np.zeros((5, 5, 3), dtype='uint8')
base64png_Image = 'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAIAAAACDbGyAAAADElEQVQIHWNgoC4AAABQAAFhFZyBAAAAAElFTkSuQmCC'
base64npy_Image = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
data = [
    (
        dict(
            box=D.Box((0, 0, 1, 1)),
            boxes=D.Boxes([(0, 0, 1, 1)]),
            polygon=D.Polygon([(0, 0), (1, 0), (1, 1)]),
            polygons=D.Polygons([[(0, 0), (1, 0), (1, 1)]]),
            np_bool=np.bool_(True),
            np_float=np.float64(1),
            np_number=np.array(1),
            np_array=np.array([1, 2]),
            image=MockImage,
            dict=dict(box=D.Box((0, 0, 1, 1))),
            str='test',
            int=1,
            float=0.6,
            tuple=(1, 1),
            pow=1e10,
        ),
        dict(
            image=lambda x: D.img_to_b64str(x, D.IMGTYP.PNG),
        ),
        dict(
            box=[0, 0, 1, 1],
            boxes=[[0, 0, 1, 1]],
            polygon=[[0, 0], [1, 0], [1, 1]],
            polygons=[[[0, 0], [1, 0], [1, 1]]],
            np_bool=True,
            np_float=1.0,
            np_number=1,
            np_array=[1, 2],
            image=base64png_Image,
            dict=dict(box=[0, 0, 1, 1]),
            str='test',
            int=1,
            float=0.6,
            tuple=[1, 1],
            pow=1e10,
        )
    ),
    (
        dict(
            image=MockImage,
        ),
        dict(
            image=lambda x: D.npy_to_b64str(x),
        ),
        dict(
            image=base64npy_Image,
        )
    ),
    (
        dict(
            images=[dict(image=MockImage)],
        ),
        dict(
            image=lambda x: D.npy_to_b64str(x),
        ),
        dict(
            images=[dict(image=base64npy_Image)],
        )
    ),
]


@pytest.mark.parametrize('x,jsonable_func,expected', data)
def test_dict_to_jsonable(x, jsonable_func, expected):
    assert dict_to_jsonable(x, jsonable_func) == expected


class TestEnum(EnumCheckMixin, Enum):
    FIRST = 1
    SECOND = "two"


class TestEnumCheckMixin:

    def test_obj_to_enum_with_valid_enum_member(self):
        assert TestEnum.obj_to_enum(TestEnum.FIRST) == TestEnum.FIRST

    def test_obj_to_enum_with_valid_string(self):
        assert TestEnum.obj_to_enum("SECOND") == TestEnum.SECOND

    def test_obj_to_enum_with_valid_int(self):
        assert TestEnum.obj_to_enum(1) == TestEnum.FIRST

    def test_obj_to_enum_with_invalid_string(self):
        with pytest.raises(ValueError):
            TestEnum.obj_to_enum("INVALID")

    def test_obj_to_enum_with_invalid_int(self):
        with pytest.raises(ValueError):
            TestEnum.obj_to_enum(3)


@dataclass
class TestDataclass(DataclassCopyMixin):
    int_field: int
    list_field: List[Any]


class TestDataclassCopyMixin:

    @pytest.fixture
    def test_dataclass_instance(self):
        return TestDataclass(10, [1, 2, 3])

    def test_shallow_copy(self, test_dataclass_instance):
        copy_instance = test_dataclass_instance.__copy__()
        assert copy_instance is not test_dataclass_instance
        assert copy_instance.int_field == test_dataclass_instance.int_field
        assert copy_instance.list_field is test_dataclass_instance.list_field

    def test_deep_copy(self, test_dataclass_instance):
        deepcopy_instance = deepcopy(test_dataclass_instance)
        assert deepcopy_instance is not test_dataclass_instance
        assert deepcopy_instance.int_field == test_dataclass_instance.int_field
        assert deepcopy_instance.list_field is not test_dataclass_instance.list_field
        assert deepcopy_instance.list_field == test_dataclass_instance.list_field


@dataclass
class TestDataclassJson(DataclassToJsonMixin, DataclassCopyMixin):
    int_field: int
    list_field: List[Any]
