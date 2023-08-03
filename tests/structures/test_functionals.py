import numpy as np
import pytest

from docsaidkit import Boxes, merge_boxes, pairwise_ioa, pairwise_iou

test_functionals_error_param = [
    (
        pairwise_iou,
        ([(1, 2, 3, 4)], [(1, 2, 3, 4)]),
        TypeError,
        'Input type of boxes1 and boxes2 must be Boxes'
    ),
    (
        pairwise_iou,
        [Boxes([(1, 1, 0, 2)], "XYWH"), Boxes([(1, 1, 0, 2)], "XYWH")],
        ValueError,
        'Some boxes in Boxes has invaild value'
    ),
    (
        pairwise_ioa,
        ([(1, 2, 3, 4)], [(1, 2, 3, 4)]),
        TypeError,
        'Input type of boxes1 and boxes2 must be Boxes'
    ),
    (
        pairwise_ioa,
        [Boxes([(1, 1, 0, 2)], "XYWH"), Boxes([(1, 1, 0, 2)], "XYWH")],
        ValueError,
        'Some boxes in Boxes has invaild value'
    ),
]


@pytest.mark.parametrize('fn, test_input, error, match', test_functionals_error_param)
def test_functionals_error(fn, test_input, error, match):
    with pytest.raises(error, match=match):
        fn(*test_input)


test_pairwise_iou_param = [(
    Boxes(np.array([[10, 10, 20, 20], [15, 15, 25, 25]]), "XYXY"),
    Boxes(np.array([[10, 10, 20, 20], [15, 15, 25, 25], [25, 25, 10, 10]]), "XYWH"),
    np.array([
        [1 / 4, 1 / 28, 0],
        [1 / 4, 4 / 25, 0]
    ], dtype='float32')
)]


@pytest.mark.parametrize('boxes1, boxes2, expected', test_pairwise_iou_param)
def test_pairwise_iou(boxes1, boxes2, expected):
    assert (pairwise_iou(boxes1, boxes2) == expected).all()


test_pairwise_ioa_param = [(
    Boxes(np.array([[10, 10, 20, 20]]), "XYXY"),
    Boxes(np.array([[15, 15, 20, 20], [20, 20, 10, 10]]), "XYWH"),
    np.array([[1 / 16, 0]], dtype='float32')
)]


@pytest.mark.parametrize('boxes1, boxes2, expected', test_pairwise_ioa_param)
def test_pairwise_ioa(boxes1, boxes2, expected):
    assert (pairwise_ioa(boxes1, boxes2) == expected).all()


test_merge_boxes_input = Boxes(np.array([
    [10, 10, 10, 10],
    [15, 15, 10, 10],
    [25, 25, 10, 10]
]), "XYWH")

test_merge_boxes_param = [
    (
        0,
        np.array([
            [10, 10, 15, 15],
            [25, 25, 10, 10]
        ]),
        [[0, 1], [2]]
    ),
    (
        0.25,
        np.array([
            [10, 10, 10, 10],
            [15, 15, 10, 10],
            [25, 25, 10, 10]
        ]),
        [[0], [1], [2]]
    ),
]


@pytest.mark.parametrize('threshold, expected1, expected2', test_merge_boxes_param)
def test_merge_boxes(threshold, expected1, expected2):
    boxes, mlabel = merge_boxes(test_merge_boxes_input, threshold)
    assert (boxes.numpy() == expected1).all()
    for m, e in zip(mlabel, expected2):
        assert type(m) == type(e)
        if isinstance(m, list):
            for m_, e_ in zip(m, e):
                assert m_ == e_
        else:
            assert m == e
