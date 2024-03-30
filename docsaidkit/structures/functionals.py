from typing import Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from .boxes import Boxes
from .polygons import Polygon

__all__ = [
    'pairwise_intersection', 'pairwise_iou', 'pairwise_ioa',
    'jaccard_index', 'polygon_iou'
]


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The type of box must be Boxes.
    Args:
        boxes1, boxes2 (Boxes):
            Two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        ndarray: intersection, sized [N, M].
    """
    if not isinstance(boxes1, Boxes) or not isinstance(boxes2, Boxes):
        raise TypeError(f'Input type of boxes1 and boxes2 must be Boxes.')

    boxes1_ = boxes1.convert('XYXY').numpy()
    boxes2_ = boxes2.convert('XYXY').numpy()
    lt = np.maximum(boxes1_[:, None, :2], boxes2_[:, :2])
    rb = np.minimum(boxes1_[:, None, 2:], boxes2_[:, 2:])
    width_height = (rb - lt).clip(min=0)
    intersection = width_height.prod(2)

    return intersection


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes):
            Two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        ndarray: IoU, sized [N,M].
    """
    if not isinstance(boxes1, Boxes) or not isinstance(boxes2, Boxes):
        raise TypeError(f'Input type of boxes1 and boxes2 must be Boxes.')

    if np.any(boxes1._xywh[:, 2:] <= 0) or np.any(boxes2._xywh[:, 2:] <= 0):
        raise ValueError(
            'Some boxes in Boxes has invaild value, which width or '
            'height is smaller than zero or other unexpected reasons, '
            'try to run "drop_empty()" at first.'
        )

    area1 = boxes1.area
    area2 = boxes2.area
    inter = pairwise_intersection(boxes1, boxes2)

    return inter / (area1[:, None] + area2 - inter)


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).
    Args:
        boxes1,boxes2 (Boxes):
            Two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        ndarray: IoA, sized [N,M].
    """
    if not isinstance(boxes1, Boxes) or not isinstance(boxes2, Boxes):
        raise TypeError(f'Input type of boxes1 and boxes2 must be Boxes.')

    if np.any(boxes1._xywh[:, 2:] <= 0) or np.any(boxes2._xywh[:, 2:] <= 0):
        raise ValueError(
            'Some boxes in Boxes has invaild value, which width or '
            'height is smaller than zero or other unexpected reasons, '
            'try to run "drop_empty()" at first.'
        )

    area2 = boxes2.area
    inter = pairwise_intersection(boxes1, boxes2)

    return inter / area2


def jaccard_index(
    pred_poly: np.ndarray,
    gt_poly: np.ndarray,
    image_size: Tuple[int, int],
) -> float:
    """
    Reference : https://github.com/jchazalon/smartdoc15-ch1-eval

    Compute the Jaccard index of two polygons.
    Args:
        pred_poly (np.ndarray):
            Predicted polygon, a 4-point polygon.
        gt_poly (np.ndarray):
            Ground truth polygon, a 4-point polygon.
        image_size (tuple):
            Image size, (height, width).

    Returns:
        float: Jaccard index.
    """

    if pred_poly.shape != (4, 2) or gt_poly.shape != (4, 2):
        raise ValueError(f'Input polygon must be 4-point polygon.')

    if image_size is None:
        raise ValueError(f'Input image size must be provided.')

    pred_poly = pred_poly.astype(np.float32)
    gt_poly = gt_poly.astype(np.float32)

    height, width = image_size
    object_coord_target = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]]
    ).astype(np.float32)

    M = cv2.getPerspectiveTransform(
        gt_poly.reshape(-1, 1, 2),
        object_coord_target.reshape(-1, 1, 2)
    )

    transformed_pred_coords = cv2.perspectiveTransform(
        pred_poly.reshape(-1, 1, 2), M)

    try:
        poly_target = ShapelyPolygon(object_coord_target)
        poly_pred = ShapelyPolygon(transformed_pred_coords.reshape(-1, 2))
        poly_inter = poly_target & poly_pred

        area_target = poly_target.area
        area_test = poly_pred.area
        area_inter = poly_inter.area

        area_union = area_test + area_target - area_inter
        # Little hack to cope with float precision issues when dealing with polygons:
        #   If intersection area is close enough to target area or GT area, but slighlty >,
        #   then fix it, assuming it is due to rounding issues.
        area_min = min(area_target, area_test)
        if area_min < area_inter and area_min * 1.0000000001 > area_inter:
            area_inter = area_min

        jaccard_index = area_inter / area_union
    except:
        # 通常錯誤來自於：
        # TopologyException: Input geom 1 is invalid: Ring Self-intersection
        # 表示多邊形自己交叉了，這時候就直接給 0
        jaccard_index = 0

    return jaccard_index


def polygon_iou(poly1: Polygon, poly2: Polygon):
    """
    Compute the IoU of two polygons.
    Args:
        poly1 (Polygon): Predicted polygon.
        poly2 (Polygon): Ground truth polygon.

    Returns:
        float: IoU.
    """
    if not isinstance(poly1, Polygon) or not isinstance(poly2, Polygon):
        raise TypeError(f'Input type of poly1 and poly2 must be Polygon.')

    poly1 = poly1.numpy().astype(np.float32)
    poly2 = poly2.numpy().astype(np.float32)

    try:
        poly1 = ShapelyPolygon(poly1)
        poly2 = ShapelyPolygon(poly2)
        poly_inter = poly1 & poly2

        area_target = poly1.area
        area_test = poly2.area
        area_inter = poly_inter.area

        area_union = area_test + area_target - area_inter
        # Little hack to cope with float precision issues when dealing with polygons:
        #   If intersection area is close enough to target area or GT area, but slighlty >,
        #   then fix it, assuming it is due to rounding issues.
        area_min = min(area_target, area_test)
        if area_min < area_inter and area_min * 1.0000000001 > area_inter:
            area_inter = area_min

        iou = area_inter / area_union
    except:
        # 通常錯誤來自於：
        # TopologyException: Input geom 1 is invalid: Ring Self-intersection
        # 表示多邊形自己交叉了，這時候就直接給 0
        iou = 0

    return iou
