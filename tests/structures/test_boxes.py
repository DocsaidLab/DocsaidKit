import numpy as np
import pytest

from docsaidkit import Box, Boxes, BoxMode


def test_invalid_input_type():
    with pytest.raises(TypeError):
        Box("invalid_input")

def test_invalid_input_shape():
    with pytest.raises(TypeError):
        Box([1, 2, 3, 4, 5])  # 長度為5而非4，不符合預期的box格式

def test_normalized_array():
    array = np.array([0.1, 0.2, 0.3, 0.4])
    box = Box(array, normalized=True)
    assert box.normalized is True

def test_invalid_box_mode():
    with pytest.raises(KeyError):
        array = np.array([1, 2, 3, 4])
        Box(array, box_mode="invalid_mode")

def test_array_conversion():
    array = [1, 2, 3, 4]
    box = Box(array)
    assert np.allclose(box._array, np.array(array, dtype='float32'))

# Test Box initialization
def test_box_init():
    # Create a box in XYXY format
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    assert isinstance(box, Box), "Initialization of Box failed."

# Test conversion of Box format
def test_box_convert():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    converted_box = box.convert(BoxMode.XYWH)
    assert np.allclose(converted_box.numpy(), np.array([50, 50, 50, 50])), "Box conversion failed."

# Test calculation of area of Box
def test_box_area():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    assert box.area == 2500, "Box area calculation failed."

# Test Box.copy() method
def test_box_copy():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    copied_box = box.copy()
    assert copied_box is not box and (copied_box._array == box._array).all(), "Box copy failed."

# Test Box conversion to numpy array
def test_box_numpy():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    arr = box.numpy()
    assert isinstance(arr, np.ndarray) and np.allclose(arr, np.array([50, 50, 100, 100])), "Box to numpy conversion failed."

# Test Box normalization
def test_box_normalize():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    normalized_box = box.normalize(200, 200)
    assert np.allclose(normalized_box.numpy(), np.array([0.25, 0.25, 0.5, 0.5])), "Box normalization failed."

# Test Box denormalization
def test_box_denormalize():
    box = Box((0.25, 0.25, 0.5, 0.5), box_mode=BoxMode.XYXY, normalized=True)
    denormalized_box = box.denormalize(200, 200)
    assert np.allclose(denormalized_box.numpy(), np.array([50, 50, 100, 100])), "Box denormalization failed."

# Test Box clipping
def test_box_clip():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    clipped_box = box.clip(60, 60, 90, 90)
    assert np.allclose(clipped_box.numpy(), np.array([60, 60, 90, 90])), "Box clipping failed."

# Test Box shifting
def test_box_shift():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    shifted_box = box.shift(10, -10)
    assert np.allclose(shifted_box.numpy(), np.array([60, 40, 110, 90])), "Box shifting failed."

# Test Box scaling
def test_box_scale():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    scaled_box = box.scale(dsize=(20, 0))
    assert np.allclose(scaled_box.numpy(), np.array([40, 50, 110, 100])), "Box scaling failed."

# Test Box to_list
def test_box_to_list():
    box = Box((50, 50, 50, 50), box_mode=BoxMode.XYXY)
    assert box.to_list() == [50, 50, 50, 50], "Boxes tolist failed."

# Test Box to_polygon
def test_box_to_polygon():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    polygon = box.to_polygon()
    assert np.allclose(polygon.numpy(), np.array([[50, 50], [100, 50], [100, 100], [50, 100]])), "Box convert_to_polygon failed."

# Test Boxes initialization
def test_invalid_input_type():
    with pytest.raises(TypeError):
        Boxes("invalid_input")

def test_invalid_input_shape():
    with pytest.raises(TypeError):
        Boxes([[1, 2, 3, 4, 5]])

def test_normalized_array():
    array = np.array([0.1, 0.2, 0.3, 0.4])
    box = Boxes([array], normalized=True)
    assert box.normalized is True

def test_invalid_box_mode():
    with pytest.raises(KeyError):
        array = np.array([1, 2, 3, 4])
        Box(array, box_mode="invalid_mode")

def test_array_conversion():
    array = [[1, 2, 3, 4]]
    box = Boxes(array)
    assert np.allclose(box._array, np.array(array, dtype='float32'))

def test_boxes_init():
    # Create boxes in XYXY format
    boxes = Boxes([(50, 50, 100, 100), [60, 60, 120, 120]], box_mode=BoxMode.XYXY)
    assert isinstance(boxes, Boxes), "Initialization of Boxes failed."

# Test conversion of Boxes format
def test_boxes_convert():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    converted_boxes = boxes.convert(BoxMode.XYWH)
    assert np.allclose(converted_boxes.numpy(), np.array([[50, 50, 50, 50], [60, 60, 60, 60]])), "Boxes conversion failed."

# Test calculation of area of Boxes
def test_boxes_area():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    assert np.allclose(boxes.area, np.array([2500, 3600])), "Boxes area calculation failed."

# Test Boxes.copy() method
def test_boxes_copy():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    copied_boxes = boxes.copy()
    assert copied_boxes is not boxes and (copied_boxes._array == boxes._array).all(), "Boxes copy failed."

# Test Boxes conversion to numpy array
def test_boxes_numpy():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    arr = boxes.numpy()
    assert isinstance(arr, np.ndarray) and np.allclose(arr, np.array([(50, 50, 100, 100), (60, 60, 120, 120)])), "Boxes to numpy conversion failed."

# Test Boxes normalization
def test_boxes_normalize():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    normalized_boxes = boxes.normalize(200, 200)
    assert np.allclose(normalized_boxes.numpy(), np.array([[0.25, 0.25, 0.5, 0.5], [0.3, 0.3, 0.6, 0.6]])), "Boxes normalization failed."

# Test Boxes denormalization
def test_boxes_denormalize():
    boxes = Boxes([(0.25, 0.25, 0.5, 0.5), (0.3, 0.3, 0.6, 0.6)], box_mode=BoxMode.XYXY, normalized=True)
    denormalized_boxes = boxes.denormalize(200, 200)
    assert np.allclose(denormalized_boxes.numpy(), np.array([(50, 50, 100, 100), (60, 60, 120, 120)])), "Boxes denormalization failed."

# Test Boxes clipping
def test_boxes_clip():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    clipped_boxes = boxes.clip(60, 60, 90, 90)
    assert np.allclose(clipped_boxes.numpy(), np.array([(60, 60, 90, 90), (60, 60, 90, 90)])), "Boxes clipping failed."

# Test Boxes shifting
def test_boxes_shift():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    shifted_boxes = boxes.shift(10, -10)
    assert np.allclose(shifted_boxes.numpy(), np.array([(60, 40, 110, 90), (70, 50, 130, 110)])), "Boxes shifting failed."

# Test Boxes scaling
def test_boxes_scale():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    scaled_boxes = boxes.scale(dsize=(20, 0))
    assert np.allclose(scaled_boxes.numpy(), np.array([(40, 50, 110, 100), (50, 60, 130, 120)])), "Boxes scaling failed."

# Test Boxes get_empty_index
def test_boxes_get_empty_index():
    boxes = Boxes([(50, 50, 50, 50), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    assert boxes.get_empty_index() == 0, "Boxes get_empty_index failed."

# Test Boxes drop_empty
def test_boxes_drop_empty():
    boxes = Boxes([(50, 50, 50, 50), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    boxes = boxes.drop_empty()
    assert np.allclose(boxes.numpy(), np.array([(60, 60, 120, 120)])), "Boxes drop_empty failed."

# Test Boxes tolist
def test_boxes_tolist():
    boxes = Boxes([(50, 50, 50, 50), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    assert boxes.tolist() == [[50, 50, 50, 50], [60, 60, 120, 120]], "Boxes tolist failed."

# Test Boxes to_polygons
def test_boxes_to_polygons():
    boxes = Boxes([(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    polygons = boxes.to_polygons()
    assert np.allclose(polygons.numpy(), np.array([[[50, 50], [100, 50], [100, 100], [50, 100]], [[60, 60], [120, 60], [120, 120], [60, 120]]])), "Boxes convert_to_polygons failed."
