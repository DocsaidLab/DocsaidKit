import numpy as np
import pytest

from docsaidkit import Polygon, Polygons, order_points_clockwise


def assert_array_equal(actual, expected):
    assert np.array_equal(actual, expected)


def assert_almost_equal(actual, expected, tolerance=1e-5):
    assert abs(actual - expected) < tolerance


tl, bl, br, tr = (0, 0), (0, 1), (1, 1), (1, 0)
POINTS_SET = np.array([(tl, bl, br, tr),
                       (tl, bl, tr, br),
                       (tl, br, bl, tr),
                       (tl, br, tr, bl),
                       (tl, tr, bl, br),
                       (tl, tr, br, bl),
                       (bl, tl, br, tr),
                       (bl, tl, tr, br),
                       (bl, br, tl, tr),
                       (bl, br, tr, tl),
                       (bl, tr, tl, br),
                       (bl, tr, br, tl),
                       (br, tl, bl, tr),
                       (br, tl, tr, bl),
                       (br, bl, tl, tr),
                       (br, bl, tr, tl),
                       (br, tr, tl, bl),
                       (br, tr, bl, tl),
                       (tr, tl, bl, br),
                       (tr, tl, br, bl),
                       (tr, bl, tl, br),
                       (tr, bl, br, tl),
                       (tr, br, tl, bl),
                       (tr, br, bl, tl)])
CLOCKWISE_PTS = np.array([tl, tr, br, bl])
COUNTER_CLOCKWISE_PTS = np.array([tl, bl, br, tr])


@pytest.mark.parametrize("pts, expected", [
    (pts, CLOCKWISE_PTS) for pts in POINTS_SET
])
def test_order_points_clockwise(pts, expected):
    ordered_pts = order_points_clockwise(pts)
    np.testing.assert_allclose(ordered_pts, expected)

@pytest.mark.parametrize("pts, expected", [
    (pts, COUNTER_CLOCKWISE_PTS) for pts in POINTS_SET
])
def test_order_points_counter_clockwise(pts, expected):
    ordered_pts = order_points_clockwise(pts, inverse=True)
    np.testing.assert_allclose(ordered_pts, expected)

def test_polygon_init():
    # Test initialization with valid arrays
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly1 = Polygon(array1)
    assert_array_equal(poly1._array, array1)

    array2 = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    poly2 = Polygon(array2)
    assert_array_equal(poly2._array, np.array(array2))

    array3 = Polygon(array1)  # Initialization with another Polygon instance
    poly3 = Polygon(array3)
    assert_array_equal(poly3._array, array1)

    array4 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    poly4 = Polygon(array4)
    assert_array_equal(poly4._array, np.array(array4))

def test_polygon_repr():
    # Test __repr__ method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)
    assert repr(poly) == f"Polygon({str(array)})"

def test_polygon_len():
    # Test __len__ method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)
    assert len(poly) == 3

def test_polygon_getitem():
    # Test __getitem__ method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)
    assert_array_equal(poly[0], np.array([1.0, 2.0]))
    assert_array_equal(poly[1], np.array([3.0, 4.0]))
    assert_array_equal(poly[2], np.array([5.0, 6.0]))

def test_polygon_normalized():
    # Test initialization with normalized=True
    array = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    poly = Polygon(array, normalized=True)
    assert_array_equal(poly._array, array.astype('float32'))
    assert poly.normalized

def test_polygon_invalid_array():
    # Test initialization with invalid arrays
    with pytest.raises(TypeError):
        invalid_array = "invalid"  # Invalid type
        Polygon(invalid_array)

    with pytest.raises(TypeError):
        invalid_array = np.array([1.0, 2.0, 3.0])  # Invalid shape
        Polygon(invalid_array)

    with pytest.raises(TypeError):
        invalid_array = [1, 2, 3]  # Invalid type
        Polygon(invalid_array)

    with pytest.raises(TypeError):
        invalid_array = [1, 2, 3, 4]  # Invalid type
        Polygon(invalid_array)

    with pytest.raises(TypeError):
        invalid_array = Polygon()  # Invalid type (empty Polygon instance)
        Polygon(invalid_array)

def test_polygon_copy():
    # Test copy method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)
    copied_poly = poly.copy()
    assert_array_equal(copied_poly._array, array)

def test_polygon_numpy():
    # Test numpy method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)
    arr = poly.numpy()
    assert_array_equal(arr, array)

def test_polygon_normalize():
    # Test normalize method
    array = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    poly = Polygon(array)
    normalized_poly = poly.normalize(100.0, 200.0)
    expected_normalized_array = np.array([[0.1, 0.1], [0.3, 0.2], [0.5, 0.3]]).astype('float32')
    assert_array_equal(normalized_poly._array, expected_normalized_array)
    assert normalized_poly.normalized

def test_polygon_denormalize():
    # Test denormalize method
    normalized_array = np.array([[0.1, 0.1], [0.3, 0.2], [0.5, 0.3]])
    poly = Polygon(normalized_array, normalized=True)
    denormalized_poly = poly.denormalize(100.0, 200.0)
    expected_denormalized_array = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]).astype('float32')
    np.testing.assert_allclose(denormalized_poly._array, expected_denormalized_array)
    assert not denormalized_poly.normalized

def test_polygon_denormalize_non_normalized():
    # Test denormalize method for non-normalized Polygon
    array = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    poly = Polygon(array)
    denormalized_poly = poly.denormalize(100.0, 200.0)
    assert not np.array_equal(denormalized_poly._array, array)
    assert not denormalized_poly.normalized

def test_polygon_clip():
    # Test clip method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)

    # Test clipping within the range
    clipped_poly = poly.clip(2, 3, 4, 5)
    expected_clipped_array = np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    assert_array_equal(clipped_poly._array, expected_clipped_array)

    # Test clipping outside the range
    clipped_poly = poly.clip(10, 20, 30, 40)
    expected_clipped_array = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    assert_array_equal(clipped_poly._array, expected_clipped_array)

def test_polygon_shift():
    # Test shift method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)

    # Test positive shift
    shifted_poly = poly.shift(2.0, 3.0)
    expected_shifted_array = np.array([[3.0, 5.0], [5.0, 7.0], [7.0, 9.0]])
    assert_array_equal(shifted_poly._array, expected_shifted_array)

    # Test negative shift
    shifted_poly = poly.shift(-2.0, -3.0)
    expected_shifted_array = np.array([[-1.0, -1.0], [1.0, 1.0], [3.0, 3.0]])
    assert_array_equal(shifted_poly._array, expected_shifted_array)

def test_polygon_scale():
    # Test scale method
    array = np.array([[10, 10], [10, 20], [20, 20], [20, 10]])
    poly = Polygon(array)

    # Test scaling with distance=1 and default join_style (mitre)
    scaled_poly = poly.scale(1)
    expected_scaled_array = np.array([[9, 9], [9, 21], [21, 21], [21, 9]])
    assert_array_equal(scaled_poly._array, expected_scaled_array)

    # Test scaling with distance=2 and round join_style
    scaled_poly = poly.scale(2, join_style=2)  # JOIN_STYLE.mitre is 2
    expected_scaled_array = np.array([[8, 8], [8, 22], [22, 22], [22, 8]])
    assert_array_equal(scaled_poly._array, expected_scaled_array)

    # Test scaling with distance=3 and bevel join_style
    scaled_poly = poly.scale(3, join_style=3)  # JOIN_STYLE.bevel is 3
    expected_scaled_array = np.array([[10, 7], [10, 23], [23, 20], [20, 7]])
    assert_array_equal(scaled_poly._array, expected_scaled_array)

def test_polygon_scale_empty():
    # Test scale method with an empty Polygon
    with pytest.raises(ValueError):
        array = np.array([])
        poly = Polygon(array)
        scaled_poly = poly.scale(1)

def test_polygon_to_convexhull():
    # Test to_convexhull method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)

    # Test convex hull of the polygon
    convex_hull_poly = poly.to_convexhull()
    expected_convex_hull_array = np.array([[5.0, 6.0], [1.0, 2.0]])
    assert_array_equal(convex_hull_poly._array, expected_convex_hull_array)

def test_polygon_to_min_boxpoints():
    # Test to_min_boxpoints method
    array = np.array([[10, 7], [10, 23], [23, 20], [20, 7]])
    poly = Polygon(array)

    # Test minimum area bounding box of the polygon
    min_box_poly = poly.to_min_boxpoints()
    expected_min_box_array = np.array([[10, 7], [23, 7], [23, 23], [10, 23]])
    assert_array_equal(min_box_poly._array, expected_min_box_array)

def test_polygon_to_box():
    # Test to_box method
    array = np.array([[10, 7], [10, 23], [23, 20], [20, 7]])
    poly = Polygon(array)

    # Test bounding box of the polygon in "xyxy" format
    box = poly.to_box(box_mode='xyxy')
    expected_box = [10, 7, 23, 23]
    assert box.tolist() == expected_box

def test_polygon_to_list():
    # Test to_list method
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)

    # Test to_list without flatten
    lst = poly.to_list()
    expected_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert lst == expected_list

    # Test to_list with flatten
    flattened_lst = poly.to_list(flatten=True)
    expected_flattened_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert flattened_lst == expected_flattened_list

def test_polygon_tolist():
    # Test tolist method (alias of to_list)
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly = Polygon(array)

    # Test tolist without flatten
    lst = poly.tolist()
    expected_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert lst == expected_list

    # Test tolist with flatten
    flattened_lst = poly.tolist(flatten=True)
    expected_flattened_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert flattened_lst == expected_flattened_list

def test_polygon_is_empty():
    # Test is_empty method
    empty_poly = Polygon([])
    non_empty_poly = Polygon([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert empty_poly.is_empty() is True
    assert non_empty_poly.is_empty() is False

def test_polygon_is_empty_with_threshold():
    # Test is_empty method with custom threshold
    empty_poly = Polygon([])
    non_empty_poly = Polygon([[1.0, 2.0], [3.0, 4.0]])

    assert empty_poly.is_empty(threshold=1) is False
    assert non_empty_poly.is_empty(threshold=3) is True


test_props_input = np.array([
    [5, 0],
    [10, 5],
    [5, 10],
    [0, 5]
])

test_props_params = [
    (
        "moments",
        {
            'm00': 50.0,
            'm10': 250.0,
            'm01': 250.0,
            'm20': 1458.3333333333333,
            'm11': 1250.0,
            'm02': 1458.3333333333333,
            'm30': 9375.0,
            'm21': 7291.666666666667,
            'm12': 7291.666666666667,
            'm03': 9375.0,
        },
    ),
    ("area", 50),
    ("arclength", 28.28427),
    ("centroid", (5, 5)),
    ("boundingbox", (0, 0, 10, 10)),
    ("min_circle", ((5.0, 5.0), 5.0)),
    ("min_box", ((5.0, 5.0), (7.07106, 7.07106), 45.0)),
    ("orientation", 45),
    ("min_box_wh", (7.07106, 7.07106)),
    ("extent", 0.5),
    ("solidity", 1.0),
]


@ pytest.mark.parametrize('prop, expected', test_props_params)
def test_polygon_property(prop, expected):
    value = getattr(Polygon(test_props_input), prop)
    if isinstance(value, (int, float)):
        np.testing.assert_allclose(value, expected, rtol=1e-4)
    elif isinstance(value, dict):
        for k, v in expected.items():
            np.testing.assert_allclose(value[k], v, rtol=1e-4)
    elif isinstance(value, (list, tuple)):
        for v, e in zip(value, expected):
            np.testing.assert_allclose(v, e, rtol=1e-4)

def test_polygons_init():
    # Test Polygons initialization
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    # Test initialization with list of arrays
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    assert len(polygons) == 2
    assert_array_equal(polygons[0]._array, array1)
    assert_array_equal(polygons[1]._array, array2)

    # Test initialization with numpy array
    polygons_array = np.array([array1, array2])
    polygons = Polygons(polygons_array)

    assert len(polygons) == 2
    assert_array_equal(polygons[0]._array, array1)
    assert_array_equal(polygons[1]._array, array2)

    # Test initialization with invalid input

    with pytest.raises(TypeError):
        invalid_input = "invalid"
        polygons = Polygons(invalid_input)

def test_polygons_len():
    # Test __len__ method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    assert len(polygons) == 2

def test_polygons_getitem():
    # Test __getitem__ method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    # Test indexing with int
    polygon = polygons[0]
    assert_array_equal(polygon._array, array1)

    # Test slicing
    sliced_polygons = polygons[0:1]
    assert len(sliced_polygons) == 1
    assert_array_equal(sliced_polygons[0]._array, array1)

    # Test indexing with list
    polygon_indices = [0, 1]
    indexed_polygons = polygons[polygon_indices]
    assert len(indexed_polygons) == 2
    assert_array_equal(indexed_polygons[0]._array, array1)
    assert_array_equal(indexed_polygons[1]._array, array2)

    # Test indexing with ndarray
    mask = np.array([True, False])
    masked_polygons = polygons[mask]
    assert len(masked_polygons) == 1
    assert_array_equal(masked_polygons[0]._array, array1)

    # Test invalid input
    with pytest.raises(TypeError):
        invalid_input = 1.5
        polygons[invalid_input]

def test_polygons_is_empty():
    # Test is_empty method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    is_empty_result = polygons.is_empty()
    expected_is_empty_result = np.array([False, True])
    assert (is_empty_result == expected_is_empty_result).all()

def test_polygons_to_min_boxpoints():
    # Test to_min_boxpoints method
    array1 = np.array([[5, 0], [10, 5], [5, 10], [0, 5]])
    array2 = np.array([[50, 0], [100, 50], [50, 100], [0, 50]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)
    min_boxpoints_polygons = polygons.to_min_boxpoints()
    assert len(min_boxpoints_polygons) == 2
    assert_array_equal(min_boxpoints_polygons[0]._array, np.array([[5, 0], [10, 5], [5, 10], [0, 5]]))
    assert_array_equal(min_boxpoints_polygons[1]._array, np.array([[50, 0], [100, 50], [50, 100], [0, 50]]))

def test_polygons_to_convexhull():
    # Test to_convexhull method
    array1 = np.array([[5, 0], [10, 5], [5, 10], [0, 5]])
    array2 = np.array([[50, 0], [100, 50], [50, 100], [0, 50]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    convexhull_polygons = polygons.to_convexhull()
    assert len(convexhull_polygons) == 2
    assert_array_equal(convexhull_polygons[0]._array, np.array([[5, 0], [10, 5], [5, 10], [0, 5]]))
    assert_array_equal(convexhull_polygons[1]._array, np.array([[50, 0], [100, 50], [50, 100], [0, 50]]))

def test_polygons_to_boxes():
    # Test to_boxes method
    array1 = np.array([[5, 0], [10, 5], [5, 10], [0, 5]])
    array2 = np.array([[50, 0], [100, 50], [50, 100], [0, 50]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    boxes = polygons.to_boxes()
    assert len(boxes) == 2
    assert_array_equal(boxes[0]._array, np.array([0, 0, 10, 10]))
    assert_array_equal(boxes[1]._array, np.array([0, 0, 100, 100]))

def test_polygons_drop_empty():
    # Test drop_empty method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    empty_threshold = 3
    non_empty_polygons = polygons.drop_empty(empty_threshold)
    assert len(non_empty_polygons) == 1
    assert_array_equal(non_empty_polygons[0]._array, array1)

def test_polygons_copy():
    # Test copy method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    copied_polygons = polygons.copy()
    assert len(copied_polygons) == 2
    assert_array_equal(copied_polygons[0]._array, array1)
    assert_array_equal(copied_polygons[1]._array, array2)
    assert copied_polygons is not polygons  # Check if a new object is created

def test_polygons_normalize():
    # Test normalize method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list, normalized=False)

    w, h = 10.0, 10.0
    normalized_polygons = polygons.normalize(w, h)
    assert len(normalized_polygons) == 2
    assert_array_equal(normalized_polygons[0]._array, np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype='float32'))
    assert_array_equal(normalized_polygons[1]._array, np.array([[0.7, 0.8], [0.9, 1.0]], dtype='float32'))
    assert normalized_polygons.normalized  # Check if the normalized flag is True

def test_polygons_denormalize():
    # Test denormalize method
    array1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    array2 = np.array([[0.7, 0.8], [0.9, 1.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list, normalized=True)

    w, h = 10.0, 10.0
    denormalized_polygons = polygons.denormalize(w, h)
    assert len(denormalized_polygons) == 2
    assert_array_equal(denormalized_polygons[0]._array, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert_array_equal(denormalized_polygons[1]._array, np.array([[7.0, 8.0], [9.0, 10.0]]))
    assert not denormalized_polygons.normalized  # Check if the normalized flag is False

def test_polygons_scale():
    # Test scale method
    array1 = np.array([[10, 10], [10, 20], [20, 20], [20, 10]])
    array2 = np.array([[10, 10], [10, 20], [20, 20], [20, 10]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    # Test scaling with distance=1 and default join_style (mitre)
    scaled_polygons = polygons.scale(1)
    assert len(scaled_polygons) == 2
    assert_array_equal(scaled_polygons[0]._array, np.array([[9, 9], [9, 21], [21, 21], [21, 9]]))
    assert_array_equal(scaled_polygons[1]._array, np.array([[9, 9], [9, 21], [21, 21], [21, 9]]))
    assert not scaled_polygons.is_empty().any()  # Check if no empty polygons after scaling

def test_polygons_numpy():
    # Test numpy method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    non_flattened_numpy_array = polygons.numpy(flatten=False)
    expected_non_flattened_array = np.array([array1, array2], dtype=object)
    for i in range(len(polygons)):
        assert_array_equal(non_flattened_numpy_array[i], polygons[i]._array)

def test_polygons_to_list():
    # Test to_list method
    array1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    array2 = np.array([[7.0, 8.0], [9.0, 10.0]])
    polygons_list = [array1, array2]
    polygons = Polygons(polygons_list)

    flattened_list = polygons.to_list(flatten=True)
    expected_flattened_list = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]]
    assert flattened_list == expected_flattened_list

    non_flattened_list = polygons.to_list(flatten=False)
    expected_non_flattened_list = [array1.tolist(), array2.tolist()]
    assert non_flattened_list == expected_non_flattened_list

test_ploygons_props_input = [
    np.array([[5, 0], [10, 5], [5, 10], [0, 5]]),
    np.array([[5, 0], [10, 5], [5, 10], [0, 5]]),
]


test_ploygons_props_params = [
    (
        "moments", [
            {
                'm00': 50.0,
                'm10': 250.0,
                'm01': 250.0,
            },
            {
                'm00': 50.0,
                'm10': 250.0,
                'm01': 250.0,
            },
        ]
    ),
    ("area", np.array([50, 50])),
    ("arclength", np.array([28.28427, 28.28427])),
    ("centroid", np.array([(5, 5), (5, 5)])),
    ("boundingbox", np.array([(0, 0, 10, 10), (0, 0, 10, 10)])),
    ("min_circle", [((5.0, 5.0), 5.0), ((5.0, 5.0), 5.0)]),
    ("min_box", [((5.0, 5.0), (7.07106, 7.07106), 45.0), ((5.0, 5.0), (7.07106, 7.07106), 45.0)]),
    ("orientation", np.array([45, 45])),
    ("min_box_wh", np.array([(7.07106, 7.07106), (7.07106, 7.07106)])),
    ("extent", np.array([0.5, 0.5])),
    ("solidity", np.array([1.0, 1.0])),
]


@ pytest.mark.parametrize('prop, expected', test_ploygons_props_params)
def test_polygons_property(prop, expected):
    value = getattr(Polygons(test_ploygons_props_input), prop)
    if isinstance(value, (int, float, np.ndarray)):
        np.testing.assert_allclose(value, expected, rtol=1e-4)
    elif isinstance(value, list):
        for v, e in zip(value, expected):
            if isinstance(v, (list, tuple)):
                for vv, ee in zip(v, e):
                    np.testing.assert_allclose(np.array(vv), np.array(ee), rtol=1e-4)
            elif isinstance(v, dict):
                for key, val in e.items():
                    assert v[key] == val
            else:
                np.testing.assert_allclose(np.array(v), np.array(e), rtol=1e-4)
