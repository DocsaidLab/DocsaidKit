from typing import List, Tuple, Union

import cv2
import numpy as np

from ...structures import Box, Boxes, Polygon, Polygons

__all__ = [
    'draw_box', 'draw_boxes', 'draw_polygon', 'draw_polygons',
]


_Color = Union[int, Tuple[int, int, int], np.ndarray]
_Colors = Union[_Color, List[_Color], np.ndarray]
_Thickness = Union[int, float, np.ndarray]
_Thicknesses = Union[List[_Thickness], _Thickness, np.ndarray]


def draw_box(
    img: np.ndarray,
    box: Union[Box, np.ndarray],
    color: _Color = (0, 255, 0),
    thickness: _Thickness = 2
) -> np.ndarray:
    """
    Draws a bounding box on the image.

    Args:
        img (np.ndarray):
            The image to draw on, as a numpy ndarray.
        box (Union[Box, np.ndarray]):
            The bounding box to draw, either as a Box object or as a numpy
            ndarray of the form [x1, y1, x2, y2].
        color (_Color, optional):
            The color of the box to draw. Defaults to (0, 255, 0).
        thickness (_Thickness, optional):
            The thickness of the box lines to draw. Defaults to 2.

    Returns:
        np.ndarray: The image with the drawn box, as a numpy ndarray.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if isinstance(box, Box):
        if box.normalized:
            h, w = img.shape[:2]
            box = box.denormalize(w, h)
        box = box.numpy()
    elif not isinstance(box, np.ndarray):
        box = np.array(box)
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)


def draw_boxes(
    img: np.ndarray,
    boxes: Union[Boxes, np.ndarray],
    color: _Colors = (0, 255, 0),
    thickness: _Thicknesses = 2
) -> np.ndarray:
    """
    Draws multiple bounding boxes on the image.

    Args:
        img (np.ndarray):
            The image to draw on, as a numpy ndarray.
        boxes (Union[Boxes, np.ndarray]):
            The bounding boxes to draw, either as a list of Box objects or as a
            2D numpy ndarray.
        color (_Colors, optional):
            The color of the boxes to draw. This can be a single color or a list
            of colors. Defaults to (0, 255, 0).
        thickness (_Thicknesses, optional):
            The thickness of the boxes lines to draw. This can be a single
            thickness or a list of thicknesses. Defaults to 2.

    Returns:
        np.ndarray: The image with the drawn boxes, as a numpy ndarray.
    """
    # If color is not a list, make it a list
    if not isinstance(color, list):
        color = [color] * len(boxes)

    # If thickness is not a list, make it a list
    if not isinstance(thickness, list):
        thickness = [thickness] * len(boxes)

    for box, c, t in zip(boxes, color, thickness):
        draw_box(img, box, color=c, thickness=t)

    return img


def draw_polygon(
    img: np.ndarray,
    polygon: Union[Polygon, np.ndarray],
    color: _Color = (0, 255, 0),
    thickness: _Thickness = 2,
    fillup=False,
    **kwargs
):
    """
    Draw a polygon on the input image.

    Args:
        img (np.ndarray):
            The input image on which the polygon will be drawn.
        polygon (Union[Polygon, np.ndarray]):
            The points of the polygon. It can be either a list of points in the
            format [(x1, y1), (x2, y2), ...] or a Polygon object.
        color (Tuple[int, int, int], optional):
            The color of the polygon (BGR format).
            Defaults to (0, 255, 0) (green).
        thickness (int, optional):
            The thickness of the polygon's edges.
            Defaults to 2.
        fill (bool, optional):
            Whether to fill the polygon with the specified color.
            Defaults to False.

    Returns:
        np.ndarray: The image with the drawn polygon.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if isinstance(polygon, Polygon):
        if polygon.normalized:
            h, w = img.shape[:2]
            polygon = polygon.denormalize(w, h)
        polygon = polygon.numpy()
    elif not isinstance(polygon, np.ndarray):
        polygon = np.array(polygon)
    polygon = polygon.astype(int)

    if fillup:
        img = cv2.fillPoly(img, [polygon], color=color, **kwargs)
    else:
        img = cv2.polylines(img, [polygon], isClosed=True, color=color, \
                            thickness=thickness, **kwargs)

    return img


def draw_polygons(
    img: np.ndarray,
    polygons: Polygons,
    color: _Colors = (0, 255, 0),
    thickness: _Thicknesses = 2,
    fillup=False,
    **kwargs
):
    """
    Draw polygons on the input image.

    Args:
        img (np.ndarray):
            The input image on which the polygons will be drawn.
        polygons (List[Union[Polygon, np.ndarray]]):
            A list of polygons to draw. Each polygon can be represented either
            as a list of points in the format [(x1, y1), (x2, y2), ...] or as a
            Polygon object.
        color (_Colors, optional):
            The color(s) of the polygons in BGR format.
            If a single color is provided, it will be used for all polygons.
            If multiple colors are provided, each polygon will be drawn with the
            corresponding color.
            Defaults to (0, 255, 0) (green).
        thickness (_Thicknesses, optional):
            The thickness(es) of the polygons' edges.
            If a single thickness value is provided, it will be used for all
            polygons. If multiple thickness values are provided, each polygon
            will be drawn with the corresponding thickness.
            Defaults to 2.
        fillup (bool, optional):
            Whether to fill the polygons with the specified color(s).
            If set to True, the polygons will be filled; otherwise, only their
            edges will be drawn.
            Defaults to False.

    Returns:
        np.ndarray: The image with the drawn polygons.
    """

    # If color is not a list, make it a list
    if not isinstance(color, list):
        color = [color] * len(polygons)

    # If thickness is not a list, make it a list
    if not isinstance(thickness, list):
        thickness = [thickness] * len(polygons)

    for polygon, c, t in zip(polygons, color, thickness):
        draw_polygon(img, polygon, color=c, thickness=t, fillup=fillup, **kwargs)

    return img
