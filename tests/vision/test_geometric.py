import numpy as np
import pytest
from docsaidkit import (INTER, ROTATE, Polygon, Polygons, imresize, imrotate,
                        imrotate90, imwarp_quadrangle, imwarp_quadrangles)


def test_imresize():
    # create a dummy image
    img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

    # resize to a larger size
    resized_img = imresize(img, (400, 200), INTER.BILINEAR)
    assert resized_img.shape == (200, 400, 3)

    # resize to a smaller size
    resized_img = imresize(img, (50, 100), INTER.BILINEAR)
    assert resized_img.shape == (100, 50, 3)

    # resize with only one dimension given
    resized_img = imresize(img, (50, None), INTER.BILINEAR)
    assert resized_img.shape[1] == 50
    assert resized_img.shape[0] == 25

    resized_img = imresize(img, (None, 50), INTER.BILINEAR)
    assert resized_img.shape[0] == 50
    assert resized_img.shape[1] == 100

    # resize and return scale
    resized_img, w_scale, h_scale = imresize(img, (400, 200), INTER.BILINEAR, return_scale=True)
    assert resized_img.shape == (200, 400, 3)
    assert w_scale == 2
    assert h_scale == 2

    # test with different interpolation
    resized_img = imresize(img, (400, 200), INTER.BILINEAR)
    assert resized_img.shape == (200, 400, 3)

    resized_img = imresize(img, (400, 200), "BILINEAR")
    assert resized_img.shape == (200, 400, 3)


def test_imrotate90():
    # Create a test image
    img = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # Test rotate_code ROTATE_90_CLOCKWISE
    rotated_img = imrotate90(img, ROTATE.ROTATE_90)
    expected_rotated_img = np.array([[7, 4, 1],
                                     [8, 5, 2],
                                     [9, 6, 3]])
    assert np.array_equal(rotated_img, expected_rotated_img)

    # Test rotate_code ROTATE_180
    rotated_img = imrotate90(img, ROTATE.ROTATE_180)
    expected_rotated_img = np.array([[9, 8, 7],
                                     [6, 5, 4],
                                     [3, 2, 1]])
    assert np.array_equal(rotated_img, expected_rotated_img)

    # Test rotate_code ROTATE_270
    rotated_img = imrotate90(img, ROTATE.ROTATE_270)
    expected_rotated_img = np.array([[3, 6, 9],
                                     [2, 5, 8],
                                     [1, 4, 7]])
    assert np.array_equal(rotated_img, expected_rotated_img)


def test_imrotate_expand_false():
    # 測試不擴展的旋轉
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 旋轉90度
    rotated_img = imrotate(img, angle=90, expand=False)
    assert rotated_img.shape == (100, 100, 3)

    # 旋轉45度
    rotated_img = imrotate(img, angle=45, expand=False)
    assert rotated_img.shape == (100, 100, 3)

def test_imrotate_expand_true():
    # 測試擴展的旋轉
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 旋轉90度
    rotated_img = imrotate(img, angle=90, expand=True)
    assert rotated_img.shape[0] >= 100
    assert rotated_img.shape[1] >= 100

    # 旋轉45度
    rotated_img = imrotate(img, angle=45, expand=True)
    assert rotated_img.shape[0] >= 100
    assert rotated_img.shape[1] >= 100

def test_imrotate_invalid_input():
    # 測試不支援的邊界填充方式
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        imrotate(img, angle=90, bordertype="invalid_bordertype")

    # 測試不支援的插值方式
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        imrotate(img, angle=90, interpolation="invalid_interpolation")

# 測試用的圖像
img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

def test_imwarp_quadrangle():
    # 使用隨機生成的四個點來進行透視變換
    src_pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    polygon = Polygon(src_pts)
    warped_img = imwarp_quadrangle(img, polygon)
    assert warped_img.shape[0] >= 80
    assert warped_img.shape[1] >= 80

def test_imwarp_quadrangle_invalid_input():
    # 測試不支援的polygon類型
    with pytest.raises(TypeError):
        polygon = "invalid_polygon"
        imwarp_quadrangle(img, polygon)

    # 測試不包含四個點的polygon
    with pytest.raises(ValueError):
        polygon = np.array([[10, 10], [90, 10], [90, 90]], dtype=np.float32)
        imwarp_quadrangle(img, polygon)

def test_imwarp_quadrangles():
    # 使用隨機生成的四個點來進行透視變換
    src_pts_1 = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    src_pts_2 = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)
    polygons = Polygons([Polygon(src_pts_1), Polygon(src_pts_2)])

    warped_images = imwarp_quadrangles(img, polygons)
    assert len(warped_images) == 2
    assert warped_images[0].shape[0] >= 80
    assert warped_images[0].shape[1] >= 80
    assert warped_images[1].shape[0] >= 60
    assert warped_images[1].shape[1] >= 60

def test_imwarp_quadrangles_invalid_input():
    # 測試不支援的polygons類型
    with pytest.raises(TypeError):
        polygons = "invalid_polygons"
        imwarp_quadrangles(img, polygons)

    # 測試包含非Polygon對象的polygons
    with pytest.raises(TypeError):
        polygons = [Polygon(np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)), "invalid_polygon"]
        imwarp_quadrangles(img, polygons)
