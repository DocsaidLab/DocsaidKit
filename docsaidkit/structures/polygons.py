from copy import deepcopy
from typing import Any, List, Tuple, Union
from warnings import warn

import cv2
import numpy as np
from shapely.geometry import JOIN_STYLE, MultiPolygon
from shapely.geometry import Polygon as _Polygon_shapely

__all__ = [
    'Polygon', 'Polygons', 'order_points_clockwise', 'JOIN_STYLE',
]

_MixedNumberType = Union[np.number, int, float]

_Polygon = Union[
    np.ndarray,
    List[Tuple[_MixedNumberType, _MixedNumberType]],
    "Polygon"
]

_Polygons = Union[
    np.ndarray,
    List["Polygon"],
    List[np.ndarray],
    List[List[Tuple[_MixedNumberType, _MixedNumberType]]],
    "Polygons"
]


def order_points_clockwise(pts: np.ndarray, inverse: bool = False) -> np.ndarray:
    """ Order the 4 points clockwise.

    Args:
        pts (np.ndarray):
            Input array of shape (4, 2) representing 4 points (x, y).
        inverse (bool, optional):
            If True, returns points in counterclockwise order. Default is False.

    Raises:
        ValueError: If the input array `pts` is not of shape (4, 2).

    Returns:
        np.ndarray: An array of shape (4, 2) representing the ordered points.
    """

    if pts.shape != (4, 2):
        raise ValueError('Input array `pts` must be of shape (4, 2).')

    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    tl, bl = left_most[np.argsort(left_most[:, 1]), :]
    tr, br = right_most[np.argsort(right_most[:, 1]), :]

    if inverse:
        points = np.stack([tl, bl, br, tr])
    else:
        points = np.stack([tl, tr, br, bl])

    return points


class Polygon:
    """
    This structure stores a list of points as a Nx2 np.ndarray.
    Stores point **annotation** data. GT Instances have a `gt_points`
    property containing the x,y location of each point. This tensor
    has shape (K, 2) where K is the number of points per instance.
    """

    def __init__(
        self,
        array: _Polygon,
        normalized: bool = False
    ):
        """
        Args:
            array (Union[np.ndarray, list]): A Nx2 or Nx1x2 matrix.
        """
        self._array = self._check_valid_array(array)
        self.normalized = normalized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._array)})"

    def __len__(self) -> int:
        return self._array.shape[0]

    def __getitem__(self, item) -> float:
        return self._array[item]

    def _check_valid_array(self, array: _Polygon) -> np.ndarray:
        if isinstance(array, (list, tuple)):
            array = np.array(array, dtype='float32')
        if isinstance(array, Polygon):
            array = array.numpy()
        cond1 = isinstance(
            array, np.ndarray) and array.ndim == 3 and array.shape[1] == 1
        cond2 = isinstance(
            array, np.ndarray) and array.ndim == 2 and array.shape[1] == 2
        cond3 = isinstance(
            array, np.ndarray) and array.ndim == 1 and len(array) == 0
        cond4 = isinstance(
            array, np.ndarray) and array.ndim == 1 and len(array) == 2
        cond5 = isinstance(array, self.__class__)
        if not (cond1 or cond2 or cond3 or cond4 or cond5):
            raise TypeError(f'Input array must be {_Polygon}.')
        if cond3 or cond4:
            array = array[None]
        if cond1:
            array = np.squeeze(array, axis=1)
        return array.astype('float32')

    def copy(self) -> "Polygon":
        """ Create a copy of the Polygon object. """
        return self.__class__(self._array)

    def numpy(self) -> np.ndarray:
        """ Convert the Polygon object to a numpy array. """
        return self._array.copy()

    def normalize(self, w: float, h: float) -> "Polygon":
        """
        Normalize the polygon coordinates.

        Args:
            w: Width of the image.
            h: Height of the image.

        Returns:
            Normalized Polygon object.
        """
        if self.normalized:
            warn(f'Normalized polygon is forced to do normalization.')
        arr = self._array.copy()
        arr = arr / (w, h)
        return self.__class__(arr, normalized=True)

    def denormalize(self, w: float, h: float) -> "Polygon":
        """
        Denormalize the Polygon coordinates.

        Args:
            w: Width of the image.
            h: Height of the image.

        Returns:
            Denormalized Polygon object.
        """
        if not self.normalized:
            warn(f'Non-normalized polygon is forced to do denormalization.')
        arr = self._array.copy()
        arr = arr * (w, h)
        return self.__class__(arr, normalized=False)

    def clip(self, xmin: int, ymin: int, xmax: int, ymax: int) -> "Polygon":
        """
        Method to clip the Polygon by limiting x coordinates to the range [xmin, xmax]
        and y coordinates to the range [ymin, ymax].

        Args:
            xmin: Minimum value of x.
            ymin: Minimum value of y.
            xmax: Maximum value of x.
            ymax: Maximum value of y.

        Returns:
            Clipped Polygon object.
        """
        if not np.isfinite(self._array).all():
            raise ValueError("Polygon ndarray contains infinite or NaN!")
        arr = self._array.copy()
        arr[:, 0] = np.clip(arr[:, 0], max(xmin, 0), xmax)
        arr[:, 1] = np.clip(arr[:, 1], max(ymin, 0), ymax)
        return self.__class__(arr)

    def shift(self, shift_x: float, shift_y: float) -> "Polygon":
        """
        Method to shift the polygon.

        Args:
            shift_x: Amount to shift in the x-axis.
            shift_y: Amount to shift in the y-axis.

        Returns:
            Shifted Polygon object.
        """
        arr = self._array.copy()
        arr += (shift_x, shift_y)
        return self.__class__(arr)

    def scale(
        self,
        distance: int,
        join_style: JOIN_STYLE = JOIN_STYLE.mitre
    ) -> "Polygon":
        """
        Returns an approximate representation of all points within a given distance
        of the this geometric object.

        The styles of joins between offset segments are specified by integer values:
            1 -> (round)
            2 -> (mitre)
            3 -> (bevel)
        These values are also enumerated by the object shapely.geometry.JOIN_STYLE
        """
        poly = _Polygon_shapely(self._array).buffer(
            distance, join_style=join_style)

        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)

        if isinstance(poly, _Polygon_shapely) and not poly.exterior.is_empty:
            pts = np.zeros_like(self._array)
            for x, y in zip(*poly.exterior.xy):
                pt = np.array([x, y])
                dist = np.linalg.norm(pt - self._array, axis=1)
                pts[dist.argmin()] = pt
        else:
            pts = []

        return self.__class__(pts)

    def to_convexhull(self) -> "Polygon":
        """
        Computes the area of all the boxes.
        Returns:
            np.ndarray: a vector with areas of each box.
        """
        hull = np.squeeze(cv2.convexHull(self._array), axis=1)
        return self.__class__(hull)

    def to_min_boxpoints(self) -> "Polygon":
        """ Converts polygon to the min area bounding box. """
        min_box = cv2.boxPoints(self.min_box).round(4)
        min_box = order_points_clockwise(np.array(min_box))
        return self.__class__(min_box)

    def to_box(self, box_mode: str = 'xyxy'):
        """ Converts polygon to the bounding box. """
        from .boxes import Box
        return Box(self.boundingbox, "xywh", self.normalized).convert(box_mode)

    def to_list(self, flatten: bool = False) -> list:
        if flatten:
            return self._array.flatten().tolist()
        else:
            return self._array.tolist()

    def tolist(self, flatten: bool = False) -> list:
        """ Alias of `to_list` (numpy style) """
        return self.to_list(flatten=flatten)

    def is_empty(self, threshold: int = 3) -> bool:
        """
        When the number of vertices is less than threshold, it cannot be
        regarded as a polygon. Defaults to 3.
        """
        if not isinstance(threshold, int):
            raise TypeError(
                f'Input threshold type error, expected "int", got "{type(threshold)}".')
        return len(self) < threshold

    @property
    def moments(self) -> dict:
        """ Get the moment of area. """
        return cv2.moments(self._array)

    @property
    def area(self) -> float:
        """ Get the region area. """
        return self.moments['m00']

    @property
    def arclength(self) -> float:
        """ Get the region arc length. """
        return cv2.arcLength(self._array, closed=True)

    @property
    def centroid(self) -> np.ndarray:
        """ Get the mass centers. """
        return np.array([
            self.moments['m10'] / (self.moments['m00'] + 1e-5),
            self.moments['m01'] / (self.moments['m00'] + 1e-5)
        ])

    @property
    def boundingbox(self) -> np.ndarray:
        """ Get the bounding box. """
        from .boxes import Box
        bbox = cv2.boundingRect(self._array)
        if not self.normalized:
            bbox = bbox - np.array([0, 0, 1, 1])
        return Box(bbox, 'xywh')

    @property
    def min_circle(self) -> Tuple[Tuple[int, int], int]:
        """ Get the min closed circle. """
        return cv2.minEnclosingCircle(self._array)

    @property
    def min_box(self) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        """ Get the min area rectangle. """
        return cv2.minAreaRect(self._array)

    @property
    def orientation(self) -> float:
        """ Get the min area rectangle. """
        _, _, angle = self.min_box
        return angle

    @property
    def min_box_wh(self) -> Tuple[float, float]:
        """ Get the min area rectangle. """
        _, (w, h), _ = self.min_box
        return w, h

    @property
    def extent(self) -> float:
        """ Ratio of pixels in the region to pixels in the total bounding box. """
        _, _, w, h = self.boundingbox
        return self.area / (w * h)

    @property
    def solidity(self) -> float:
        """ Ratio of pixels in the region to pixels of the convex hull image. """
        return self.area / (self.to_convexhull().area + 1e-5)


class Polygons:

    def __init__(self, polygons: _Polygons, normalized: bool = False):
        if not isinstance(polygons, (list, np.ndarray)):
            raise TypeError(
                f'Input type error: "{polygons}", must be list or np.ndarray type.')
        self.normalized = normalized
        self._polygons = [Polygon(p, normalized) for p in polygons]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._polygons)})"

    def __len__(self) -> int:
        return len(self._polygons)

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item) -> Union["Polygons", "Polygon"]:
        """
        Args:
            item: int, slice, or a Boolndarray

        Returns:
            Polygons: Create a new :class:`Polygons` by indexing.

        Notes:
            If using a index, likes Polygons[2], will return a "Polygon";
            and if using a slice or vector, likes Polygons[0:1], will return a "Polygons".

        The following usage are allowed:
            1. `new_polygons = polygons[3]`: return a `Polygon`.
            2. `new_polygons = polygons[2:10]`: return a slice of Polygons.
            3. `new_polygons = polygons[vector]`, where vector is a np.Boolndarray
                with `length = len(polygons)`. Nonzero elements in the vector will be selected.
        """
        if isinstance(item, int):
            output = self._polygons[item]
        elif isinstance(item, list):
            output = Polygons([self._polygons[i] for i in item])
        elif isinstance(item, slice):
            output = Polygons(self._polygons[item])
        elif isinstance(item, np.ndarray):
            if item.dtype == 'bool':
                item = np.argwhere(item).flatten()
            output = Polygons([self._polygons[i] for i in item])
        else:
            raise TypeError(
                'Input item type error, expected to be int, list, ndarray or slice.')
        return output

    def is_empty(self, threshold: int = 3) -> np.ndarray:
        return np.array([poly.is_empty(threshold) for poly in self._polygons])

    def to_min_boxpoints(self) -> "Polygons":
        return Polygons([poly.to_min_boxpoints() for poly in self._polygons])

    def to_convexhull(self) -> "Polygons":
        return Polygons([poly.to_convexhull() for poly in self._polygons])

    def to_boxes(self, box_mode: str = 'xyxy'):
        from .boxes import Boxes
        return Boxes(self.boundingbox, 'xywh', self.normalized).convert(box_mode)

    def drop_empty(self, threshold: int = 3) -> "Polygons":
        return Polygons([p for p in self._polygons if not p.is_empty(threshold)])

    def copy(self):
        return self.__class__(deepcopy(self._polygons))

    def normalize(self, w: float, h: float) -> "Polygons":
        if self.normalized:
            warn(f'Normalized polygons are forced to do normalization.')
        _polygons = [x.normalize(w, h) for x in self._polygons]
        polygons = self.__class__(_polygons, normalized=True)
        return polygons

    def denormalize(self, w: float, h: float) -> "Polygons":
        if not self.normalized:
            warn(f'Non-normalized polygons are forced to do denormalization.')
        _polygons = [x.denormalize(w, h) for x in self._polygons]
        polygons = self.__class__(_polygons, normalized=False)
        return polygons

    def clip(self, xmin: int, ymin: int, xmax: int, ymax: int) -> "Polygons":
        return Polygons([p.clip(xmin, ymin, xmax, ymax) for p in self._polygons])

    def shift(self, shift_x: float, shift_y: float) -> "Polygons":
        return Polygons([p.shift(shift_x, shift_y) for p in self._polygons])

    def scale(self, distance: int) -> "Polygons":
        return Polygons([p.scale(distance) for p in self._polygons]).drop_empty()

    def numpy(self, flatten: bool = False):
        len_polys = np.array([len(p) for p in self._polygons], dtype=np.int32)
        if (len_polys == len_polys.mean()).all():
            return np.array(self.to_list(flatten=flatten)).astype('float32')
        else:
            return np.array(self._polygons, dtype=object)

    def to_list(self, flatten: bool = False) -> list:
        """ Convert boxes to list.

            Args:

            is_flatten (bool):
                True -> Output format (Nx(Mx2)):
                    [
                        [p11, p12, p13, p14],
                        [p21, p22, p23, p24],
                        ...,
                    ].
                False -> Output format (NxMx2):
                    [
                        [
                            [p11],
                            [p12],
                            [p13],
                            [p14]
                        ],
                        [
                            [p21],
                            [p22],
                            [p23],
                            [p24]
                        ],
                        ...,
                    ].
        """
        return [p.to_list(flatten) for p in self._polygons]

    def tolist(self, flatten: bool = False) -> list:
        """ Alias of `to_list` (numpy style) """
        return self.to_list(flatten=flatten)

    @ property
    def moments(self) -> list:
        return [poly.moments for poly in self._polygons]

    @ property
    def min_circle(self) -> list:
        return [poly.min_circle for poly in self._polygons]

    @ property
    def min_box(self) -> list:
        return [poly.min_box for poly in self._polygons]

    @ property
    def area(self) -> np.ndarray:
        return np.array([poly.area for poly in self._polygons])

    @ property
    def arclength(self) -> np.ndarray:
        return np.array([poly.arclength for poly in self._polygons])

    @ property
    def centroid(self) -> np.ndarray:
        return np.array([poly.centroid for poly in self._polygons])

    @property
    def boundingbox(self) -> np.ndarray:
        return np.array([poly.boundingbox for poly in self._polygons])

    @ property
    def extent(self) -> np.ndarray:
        return np.array([poly.extent for poly in self._polygons])

    @ property
    def solidity(self) -> np.ndarray:
        return np.array([poly.solidity for poly in self._polygons])

    @ property
    def orientation(self) -> np.ndarray:
        return np.array([poly.orientation for poly in self._polygons])

    @property
    def min_box_wh(self) -> np.ndarray:
        return np.array([poly.min_box_wh for poly in self._polygons])

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        mode: int = cv2.RETR_EXTERNAL,
        method: int = cv2.CHAIN_APPROX_SIMPLE
    ) -> "Polygons":
        if not isinstance(image, np.ndarray):
            raise TypeError('Input image must be a np.ndarray.')
        contours, _ = cv2.findContours(image, mode=mode, method=method)
        if len(contours) > 0:
            contours = [c for c in contours if c.shape[0] > 1]
        return cls(list(contours))

    @classmethod
    def cat(cls, polygons_list: List["Polygons"]) -> "Polygons":
        """
        Concatenates a list of Polygon into a single Polygons.
        Returns:
            Polygon: the concatenated Polygon
        """
        if not isinstance(polygons_list, list):
            raise TypeError('Given polygon_list should be a list.')

        if len(polygons_list) == 0:
            raise ValueError('Given polygon_list is empty.')

        if not all(isinstance(polygons, Polygons) for polygons in polygons_list):
            raise TypeError(
                'All type of elements in polygon_list must be Polygon.')

        _polygons = []
        for polys in polygons_list:
            _polygons.extend(polys._polygons)

        return cls(_polygons)
