from typing import Tuple

import networkx as nx
import numpy as np

from .boxes import Boxes

__all__ = [
    'pairwise_intersection', 'pairwise_iou', 'pairwise_ioa', 'merge_boxes',
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


def merge_boxes(boxes: Boxes, threshold: float = 0) -> Tuple["Boxes", list]:
    """
    Function to merge the overlapping bounding boxes.
    This function uses graph theory to analyze the associated bbox,
    and "networx" package have to be installed before calling function.

    Installing package as follows:
        pip install networx

    NetworkX is a Python package for the creation, manipulation, and
    study of the structure, dynamics, and functions of complex networks.

    Args:
        bboxes (Boxes):
            Input bounding boxes.
        threshold (float):
            The overlap ratio (range in 0~1) between two bboxes larger
            than threshold will be merged.

    Returns:
        merged_boxes (Boxes): Output merged bounding boxes.
        merged_idx (bool): Output groups of merged index of bounding boxes.
    """
    ratio = pairwise_iou(boxes, boxes)
    ratio = np.tril(ratio, 0)

    # get components
    xx, yy = np.where(ratio > threshold)
    edges = [(x, y) for x, y in zip(xx, yy)]
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # merge bboxes
    mboxes, mlabel = [], []
    arr = boxes.convert('XYXY').numpy()
    for conn in nx.connected_components(graph):
        conn = list(conn)
        if arr[conn, :].shape[0] == 1:
            mboxes.append(arr[conn, :][0])
        else:
            x1 = arr[conn, :][:, 0].min()
            y1 = arr[conn, :][:, 1].min()
            x2 = arr[conn, :][:, 2].max()
            y2 = arr[conn, :][:, 3].max()
            mboxes.append(np.array((x1, y1, x2, y2)))
        mlabel.append(conn)
    mboxes = Boxes(mboxes, box_mode='XYXY').convert(boxes.box_mode)

    return mboxes, mlabel
