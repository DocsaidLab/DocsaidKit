from enum import Enum, unique
from typing import Any, List, Tuple, Union
from warnings import warn

import numpy as np

__all__ = ['BoxMode', 'Box', 'Boxes']

_BoxMode = Union["BoxMode", int, str]

_MixedNumberType = Union[np.ndarray, int, float]

_Box = Union[
    np.ndarray,
    Tuple[_MixedNumberType, _MixedNumberType,
          _MixedNumberType, _MixedNumberType],
    "Box"
]

_Boxes = Union[
    np.ndarray,
    List["Box"],
    List[np.ndarray],
    List[Tuple[_MixedNumberType, _MixedNumberType,
               _MixedNumberType, _MixedNumberType]],
    "Boxes"
]


@unique
class BoxMode(Enum):
    """
    Enum of different ways to represent a box.
    """

    XYXY = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """

    XYWH = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    (x0, y0) is the top-left point of the bounding box.
    (w, h) is width and height of the bounding box.
    """

    CXCYWH = 2
    """
    (xc, yc, w, h) in absolute floating points coordinates.
    (xc, yc) is the center of the bounding box.
    (w, h) is width and height of the bounding box.
    """

    @staticmethod
    def convert(box: np.ndarray, from_mode: _BoxMode, to_mode: _BoxMode) -> np.ndarray:
        """
        Convert function for box format converting

        Args:
            box (np.ndarray): can be a box or boxes.
            from_mode (BoxMode): BoxMode
            to_mode (BoxMode): BoxMode

        Returns:
            np.ndarray: converted boxes.
        """
        arr = box.copy()
        from_mode = BoxMode.align_code(from_mode)
        to_mode = BoxMode.align_code(to_mode)

        if from_mode == to_mode:
            pass
        elif from_mode == BoxMode.XYWH and to_mode == BoxMode.XYXY:
            arr[..., 2:] += arr[..., :2]
        elif from_mode == BoxMode.XYWH and to_mode == BoxMode.CXCYWH:
            arr[..., :2] += arr[..., 2:] / 2
        elif from_mode == BoxMode.XYXY and to_mode == BoxMode.XYWH:
            arr[..., 2:] -= arr[..., :2]
        elif from_mode == BoxMode.XYXY and to_mode == BoxMode.CXCYWH:
            arr[..., 2:] -= arr[..., :2]
            arr[..., :2] += arr[..., 2:] / 2
        elif from_mode == BoxMode.CXCYWH and to_mode == BoxMode.XYXY:
            arr[..., :2] -= arr[..., 2:] / 2
            arr[..., 2:] += arr[..., :2]
        elif from_mode == BoxMode.CXCYWH and to_mode == BoxMode.XYWH:
            arr[..., :2] -= arr[..., 2:] / 2
        else:
            raise NotImplementedError(
                f"Conversion from BoxMode {str(from_mode)} to {str(to_mode)} is not supported yet")
        return arr

    @staticmethod
    def align_code(box_mode: _BoxMode):
        if isinstance(box_mode, int):
            return BoxMode(box_mode)
        elif isinstance(box_mode, str):
            return BoxMode[box_mode.upper()]
        elif isinstance(box_mode, BoxMode):
            return box_mode
        else:
            raise TypeError(f'Given `box_mode` is not int, str, or BoxMode.')


class Box:

    def __init__(
        self,
        array: _Box,
        box_mode: _BoxMode = BoxMode.XYXY,
        normalized: bool = False
    ):
        """
        Args:
            array (_Box): A box.
            box_mode (_BoxMode):
                Enum of different ways to represent a box.
                see: BoxMode for more info.

        """
        self.box_mode = BoxMode.align_code(box_mode)
        self.normalized = normalized
        self._array = self._check_valid_array(array)
        self._xywh = BoxMode.convert(self._array, self.box_mode, BoxMode.XYWH)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._array)}), {str(BoxMode(self.box_mode))}"

    def __len__(self):
        return self._array.shape[0]

    def __getitem__(self, item) -> float:
        return self._array[item]

    def _check_valid_array(self, array: Any) -> np.ndarray:
        cond1 = isinstance(array, tuple) and len(array) == 4
        cond2 = isinstance(array, list) and len(array) == 4
        cond3 = isinstance(
            array, np.ndarray) and array.ndim == 1 and len(array) == 4
        cond4 = isinstance(array, self.__class__)
        if not (cond1 or cond2 or cond3 or cond4):
            raise TypeError(f'Input array must be {_Box}.')
        if cond3:
            array = array.astype('float32')
        elif cond4:
            array = array.numpy()
        else:
            array = np.array(array, dtype='float32')
        return array

    def convert(self, to_mode: _BoxMode) -> "Box":
        """
        Method to convert box format.

        Args:
            to_mode: Target box mode.

        Returns:
            Converted Box object.
        """
        transed = BoxMode.convert(self._array, self.box_mode, to_mode)
        return self.__class__(transed, to_mode)

    def copy(self) -> Any:
        """ Create a copy of the Box object. """
        return self.__class__(self._array, self.box_mode)

    def numpy(self) -> np.ndarray:
        """ Convert the Box object to a numpy array. """
        return self._array.copy()

    def square(self) -> "Box":
        """ Convert the box to a square box. """
        arr = self.convert('CXCYWH').numpy()
        arr[2:] = arr[2:].min()
        return self.__class__(arr, 'CXCYWH').convert(self.box_mode)

    def normalize(self, w: int, h: int) -> "Box":
        """
        Normalize the box coordinates.

        Args:
            w: Width of the image.
            h: Height of the image.

        Returns:
            Normalized Box object.
        """
        if self.normalized:
            warn(f'Normalized box is forced to do normalization.')
        arr = self._array.copy()
        arr[::2] = arr[::2] / w
        arr[1::2] = arr[1::2] / h
        return self.__class__(arr, self.box_mode, normalized=True)

    def denormalize(self, w: int, h: int) -> "Box":
        """
        Denormalize the box coordinates.

        Args:
            w: Width of the image.
            h: Height of the image.

        Returns:
            Denormalized Box object.
        """
        if not self.normalized:
            warn(f'Non-normalized box is forced to do denormalization.')
        arr = self._array.copy()
        arr[::2] = arr[::2] * w
        arr[1::2] = arr[1::2] * h
        return self.__class__(arr, self.box_mode, normalized=False)

    def clip(self, xmin: int, ymin: int, xmax: int, ymax: int) -> "Box":
        """
        Method to clip the box by limiting x coordinates to the range [xmin, xmax]
        and y coordinates to the range [ymin, ymax].

        Args:
            xmin: Minimum value of x.
            ymin: Minimum value of y.
            xmax: Maximum value of x.
            ymax: Maximum value of y.

        Returns:
            Clipped Box object.
        """
        if not np.isfinite(self._array).all():
            raise ValueError("Box ndarray contains infinite or NaN!")

        arr = BoxMode.convert(self._array, self.box_mode, BoxMode.XYXY)
        arr[0::2] = np.clip(arr[0::2], max(xmin, 0), xmax)
        arr[1::2] = np.clip(arr[1::2], max(ymin, 0), ymax)
        return self.__class__(arr, self.box_mode)

    def shift(self, shift_x: float, shift_y: float) -> "Box":
        """
        Method to shift the box.

        Args:
            shift_x: Amount to shift in the x-axis.
            shift_y: Amount to shift in the y-axis.

        Returns:
            Shifted Box object.
        """
        arr = self._xywh.copy()
        arr[:2] += (shift_x, shift_y)
        return self.__class__(arr, "XYWH").convert(self.box_mode)

    def scale(self, dsize: Tuple[int, int] = None, fx: float = None, fy: float = None) -> "Box":
        """
        Method to scale Box with a given scale.

        Args:
            dsize: (Tuple[int, int]):
                Expand width and height of Box.
            fx: (Union[int, float]):
                Scaling ratio larger than 0 is to shrink or expand width of Box.
            fy: (Union[int, float]):
                Scaling ratio larger than 0 is to shrink or expand height of Box.

            Using Eg:
                1. int type: Box will be expanded or shrunk with absolute size.
                    scale a Box([x, y, w, h]) with dsize = (10, 0) -> Box([x-5, y, w+10, h])
                2. float tpye: Box will be expanded or shrunk with a relative ratio.
                    scale a Box([x, y, w, h]) with fx = 1.1 ->
                        Box([x - int(0.05 * w), y, int(1.1 * w), h])

        Returns:
            A scaled Box object.
        """
        arr = self._xywh.copy()

        if dsize is not None:
            dx, dy = dsize
            arr[0] -= dx / 2
            arr[2] += dx
            arr[1] -= dy / 2
            arr[3] += dy
        else:
            if fx is not None:
                fx = arr[2] * (fx - 1)
                arr[0] -= fx / 2
                arr[2] += fx
            if fy is not None:
                fy = arr[3] * (fy - 1)
                arr[1] -= fy / 2
                arr[3] += fy

        return self.__class__(arr, "XYWH").convert(self.box_mode)

    def to_list(self) -> list:
        return self._array.tolist()

    def tolist(self) -> list:
        """ Alias of `to_list` (numpy style) """
        return self.to_list()

    def to_polygon(self):
        from .polygons import Polygon
        arr = self._xywh.copy()
        if (arr[2:] <= 0).any():
            raise ValueError(
                'Some element in Box has invaild value, which width or '
                'height is smaller than zero or other unexpected reasons.'
            )
        p1 = arr[:2]
        p2 = np.stack([arr[0::2].sum(), arr[1]])
        p3 = arr[:2] + arr[2:]
        p4 = np.stack([arr[0], arr[1::2].sum()])
        return Polygon(np.stack([p1, p2, p3, p4]), self.normalized)

    @ property
    def width(self) -> np.ndarray:
        """ Get width of the box. """
        return self._xywh[2]

    @ property
    def height(self) -> np.ndarray:
        """ Get height of the box. """
        return self._xywh[3]

    @ property
    def left_top(self) -> np.ndarray:
        """ Get the left-top point of the box. """
        return self._xywh[0:2]

    @ property
    def right_bottom(self) -> np.ndarray:
        """ Get the right-bottom point of the box. """
        return self._xywh[0:2] + self._xywh[2:4]

    @ property
    def left_bottom(self) -> np.ndarray:
        """ Get the left_bottom point of the box. """
        return self._xywh[0:2] + [0, self._xywh[3]]

    @ property
    def right_top(self) -> np.ndarray:
        """ Get the right_top point of the box. """
        return self._xywh[0:2] + [self._xywh[2], 0]

    @ property
    def area(self) -> np.ndarray:
        """ Get the area of the boxes. """
        return self._xywh[2] * self._xywh[3]

    @ property
    def aspect_ratio(self) -> np.ndarray:
        """ Compute the aspect ratios (widths / heights) of the boxes. """
        return self._xywh[2] / self._xywh[3]

    @ property
    def center(self) -> np.ndarray:
        """ Compute the center of the box. """
        return self._xywh[:2] + self._xywh[2:] / 2


class Boxes:

    def __init__(
        self,
        array: _Boxes,
        box_mode: _BoxMode = BoxMode.XYXY,
        normalized: bool = False
    ):
        self.box_mode = BoxMode.align_code(box_mode)
        self.normalized = normalized
        self._array = self._check_valid_array(array)
        self._xywh = BoxMode.convert(self._array, box_mode, BoxMode.XYWH)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._array)}), {str(BoxMode(self.box_mode))}"

    def __len__(self):
        return self._array.shape[0]

    def __getitem__(self, item) -> Union["Box", "Boxes"]:
        if isinstance(item, int):
            return Box(self._array[item], self.box_mode, normalized=self.normalized)
        if isinstance(item, (list, slice, np.ndarray)):
            return self.__class__(self._array[item], self.box_mode, normalized=self.normalized)

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]

    def _check_valid_array(self, array: Any) -> np.ndarray:
        cond1 = isinstance(array, list)
        cond2 = isinstance(
            array, np.ndarray) and array.ndim == 2 and array.shape[-1] == 4
        cond3 = isinstance(
            array, np.ndarray) and array.ndim == 1 and len(array) == 0
        cond4 = isinstance(array, self.__class__)
        if not (cond1 or cond2 or cond3 or cond4):
            raise TypeError(f'Input array must be {_Boxes}.')
        if cond1:
            for i, x in enumerate(array):
                try:
                    array[i] = Box(x, box_mode=self.box_mode,
                                   normalized=self.normalized).numpy()
                except TypeError:
                    raise TypeError(f'Input array[{i}] must be {_Box}.')
        if cond4:
            array = [box.convert(self.box_mode).numpy() for box in array]
        array = np.array(array, dtype='float32').copy()
        return array

    def convert(self, to_mode: _BoxMode) -> "Boxes":
        """
        Method to convert box format.

        Args:
            to_mode: Target box mode.

        Returns:
            Converted Box object.
        """
        transed = BoxMode.convert(self._array, self.box_mode, to_mode)
        return self.__class__(transed, to_mode)

    def copy(self) -> Any:
        """ Create a copy of the Box object. """
        return self.__class__(self._array, self.box_mode)

    def numpy(self) -> np.ndarray:
        """ Convert the Box object to a numpy array. """
        return self._array.copy()

    def square(self) -> "Boxes":
        arr = self.convert('CXCYWH').numpy()
        arr[..., 2:] = arr[..., 2:].max(1)
        return self.__class__(arr, 'CXCYWH').convert(self.box_mode)

    def normalize(self, w: int, h: int) -> "Boxes":
        """
        Normalize the box coordinates.

        Args:
            w: Width of the image.
            h: Height of the image.

        Returns:
            Normalized Box object.
        """
        if self.normalized:
            warn(f'Normalized box is forced to do normalization.')
        arr = self._array.copy()
        arr[:, ::2] = arr[:, ::2] / w
        arr[:, 1::2] = arr[:, 1::2] / h
        return self.__class__(arr, self.box_mode, normalized=True)

    def denormalize(self, w: int, h: int) -> "Boxes":
        """
        Denormalize the box coordinates.

        Args:
            w: Width of the image.
            h: Height of the image.

        Returns:
            Denormalized Boxes object.
        """
        if not self.normalized:
            warn(f'Non-normalized box is forced to do denormalization.')
        arr = self._array.copy()
        arr[:, ::2] = arr[:, ::2] * w
        arr[:, 1::2] = arr[:, 1::2] * h
        return self.__class__(arr, self.box_mode, normalized=False)

    def clip(self, xmin: int, ymin: int, xmax: int, ymax: int) -> "Box":
        """
        Method to clip the box by limiting x coordinates to the range [xmin, xmax]
        and y coordinates to the range [ymin, ymax].

        Args:
            xmin: Minimum value of x.
            ymin: Minimum value of y.
            xmax: Maximum value of x.
            ymax: Maximum value of y.

        Returns:
            Clipped Boxes object.
        """
        if not np.isfinite(self._array).all():
            raise ValueError("Box ndarray contains infinite or NaN!")

        arr = BoxMode.convert(self._array, self.box_mode, BoxMode.XYXY)
        arr[:, 0::2] = np.clip(arr[:, 0::2], max(xmin, 0), xmax)
        arr[:, 1::2] = np.clip(arr[:, 1::2], max(ymin, 0), ymax)
        return self.__class__(arr, self.box_mode)

    def shift(self, shift_x: float, shift_y: float) -> "Boxes":
        """
        Method to shift the box.

        Args:
            shift_x: Amount to shift in the x-axis.
            shift_y: Amount to shift in the y-axis.

        Returns:
            Shifted Boxes object.
        """
        arr = self._xywh.copy()
        arr[:, :2] += (shift_x, shift_y)
        return self.__class__(arr, "XYWH").convert(self.box_mode)

    def scale(self, dsize: Tuple[int, int] = None, fx: float = None, fy: float = None) -> "Boxes":
        """
        Method to scale Box with a given scale.

        Args:
            dsize: (Tuple[int, int]):
                Expand width and height of Box.
            fx: (Union[int, float]):
                Scaling ratio larger than 0 is to shrink or expand width of Box.
            fy: (Union[int, float]):
                Scaling ratio larger than 0 is to shrink or expand height of Box.

            Using Eg:
                1. int type: Box will be expanded or shrunk with absolute size.
                    scale a Box([x, y, w, h]) with dsize = (10, 0) -> Box([x-5, y, w+10, h])
                2. float tpye: Box will be expanded or shrunk with a relative ratio.
                    scale a Box([x, y, w, h]) with fx = 1.1 ->
                        Box([x - int(0.05 * w), y, int(1.1 * w), h])

        Returns:
            A scaled Boxes object.
        """
        arr = self._xywh.copy()

        if dsize is not None:
            dx, dy = dsize
            arr[:, 0] -= dx / 2
            arr[:, 2] += dx
            arr[:, 1] -= dy / 2
            arr[:, 3] += dy
        else:
            if fx is not None:
                fx = arr[:, 2] * (fx - 1)
                arr[:, 0] -= fx / 2
                arr[:, 2] += fx
            if fy is not None:
                fy = arr[3] * (fy - 1)
                arr[:, 1] -= fy / 2
                arr[:, 3] += fy

        return self.__class__(arr, "XYWH").convert(self.box_mode)

    def get_empty_index(self) -> np.ndarray:
        """ Get the index of empty boxes. """
        return np.where((self._xywh[:, 2] <= 0) | (self._xywh[:, 3] <= 0))[0]

    def drop_empty(self) -> "Boxes":
        """ Drop the empty boxes. """
        return self.__class__(self._array[(self._xywh[:, 2] > 0) & (self._xywh[:, 3] > 0)], self.box_mode)

    def to_list(self) -> list:
        return self._array.tolist()

    def tolist(self) -> list:
        """ Alias of `to_list` (numpy style) """
        return self.to_list()

    def to_polygons(self):
        from .polygons import Polygons
        arr = self._xywh.copy()
        if (arr[:, 2:] <= 0).any():
            raise ValueError(
                'Some element in Boxes has invaild value, which width or '
                'height is smaller than zero or other unexpected reasons.'
            )

        p1 = arr[:, :2]
        p2 = np.stack([arr[:, 0::2].sum(1), arr[:, 1]], axis=1)
        p3 = arr[:, :2] + arr[:, 2:]
        p4 = np.stack([arr[:, 0], arr[:, 1::2].sum(1)], axis=1)
        return Polygons(np.stack([p1, p2, p3, p4], axis=1), self.normalized)

    @ property
    def width(self) -> np.ndarray:
        """ Get width of the box. """
        return self._xywh[:, 2]

    @ property
    def height(self) -> np.ndarray:
        """ Get height of the box. """
        return self._xywh[:, 3]

    @ property
    def left_top(self) -> np.ndarray:
        """ Get the left-top point of the box. """
        return self._xywh[:, :2]

    @ property
    def right_bottom(self) -> np.ndarray:
        """ Get the right-bottom point of the box. """
        return self._xywh[:, :2] + self._xywh[:, 2:4]

    @ property
    def area(self) -> np.ndarray:
        """ Get the area of the boxes. """
        return self._xywh[:, 2] * self._xywh[:, 3]

    @ property
    def aspect_ratio(self) -> np.ndarray:
        """ Compute the aspect ratios (widths / heights) of the boxes. """
        return self._xywh[:, 2] / self._xywh[:, 3]

    @ property
    def center(self) -> np.ndarray:
        """ Compute the center of the box. """
        return self._xywh[:, :2] + self._xywh[:, 2:] / 2
