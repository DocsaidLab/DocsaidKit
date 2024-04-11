import cv2
import numpy as np
import pytest
from docsaidkit import (BORDER, Box, Boxes, gaussianblur, imbinarize,
                        imcropbox, imcropboxes, imcvtcolor, meanblur,
                        medianblur, pad)


def test_meanblur():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試預設ksize
    blurred_img_default = meanblur(img)
    assert blurred_img_default.shape == img.shape

    # 測試指定ksize
    ksize = (5, 5)
    blurred_img_custom = meanblur(img, ksize=ksize)
    assert blurred_img_custom.shape == img.shape


def test_gaussianblur():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試預設ksize和sigmaX
    blurred_img_default = gaussianblur(img)
    assert blurred_img_default.shape == img.shape

    # 測試指定ksize和sigmaX
    ksize = (7, 7)
    sigmaX = 1
    blurred_img_custom = gaussianblur(img, ksize=ksize, sigmaX=sigmaX)
    assert blurred_img_custom.shape == img.shape


def test_medianblur():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試預設ksize
    blurred_img_default = medianblur(img)
    assert blurred_img_default.shape == img.shape

    # 測試指定ksize
    ksize = 5
    blurred_img_custom = medianblur(img, ksize=ksize)
    assert blurred_img_custom.shape == img.shape


def test_imcvtcolor():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試RGB轉灰階
    gray_img = imcvtcolor(img, 'RGB2GRAY')
    assert gray_img.shape == (100, 100)

    # 測試RGB轉BGR
    bgr_img = imcvtcolor(img, 'RGB2BGR')
    assert bgr_img.shape == img.shape

    # 測試轉換為不支援的色彩空間
    with pytest.raises(ValueError):
        imcvtcolor(img, 'RGB2WWW')  # XYZ為不支援的色彩空間


def test_pad_constant_gray():
    # 測試用的灰階圖片
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    # 測試等值填充
    pad_size = 10
    fill_value = 128
    padded_img = pad(img, pad_size=pad_size, fill_value=fill_value)
    assert padded_img.shape == (
        img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size)
    assert np.all(padded_img[:pad_size, :] == fill_value)
    assert np.all(padded_img[-pad_size:, :] == fill_value)
    assert np.all(padded_img[:, :pad_size] == fill_value)
    assert np.all(padded_img[:, -pad_size:] == fill_value)


def test_pad_constant_color():
    # 測試用的彩色圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試等值填充
    pad_size = 5
    fill_value = (255, 0, 0)  # 紅色
    padded_img = pad(img, pad_size=pad_size, fill_value=fill_value)
    assert padded_img.shape == (
        img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size, img.shape[2])
    assert np.all(padded_img[:pad_size, :, :] == fill_value)
    assert np.all(padded_img[-pad_size:, :, :] == fill_value)
    assert np.all(padded_img[:, :pad_size, :] == fill_value)
    assert np.all(padded_img[:, -pad_size:, :] == fill_value)


def test_pad_replicate():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試邊緣複製填充
    pad_size = (5, 10)
    padded_img = pad(img, pad_size=pad_size, pad_mode=BORDER.REPLICATE)
    assert padded_img.shape == (
        img.shape[0] + 2 * pad_size[0], img.shape[1] + 2 * pad_size[1], img.shape[2])


def test_pad_reflect():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試邊緣反射填充
    pad_size = (0, 10, 15, 5)
    padded_img = pad(img, pad_size=pad_size, pad_mode=BORDER.REFLECT)
    assert padded_img.shape == (
        img.shape[0] + pad_size[0] + pad_size[1],
        img.shape[1] + pad_size[2] + pad_size[3],
        img.shape[2]
    )


def test_pad_invalid_input():
    # 測試不支援的填充模式
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pad(img, pad_size=5, pad_mode='invalid_mode')

    # 測試不合法的填充大小
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pad(img, pad_size=(10, 20, 30))


def test_imcropbox_custom_box():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用自定義Box物件進行裁剪
    box = Box([20, 30, 80, 60])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (30, 60, 3)


def test_imcropbox_numpy_array():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用NumPy陣列進行裁剪
    box = np.array([20, 30, 80, 60])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (30, 60, 3)


def test_imcropbox_outside_boundary():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 裁剪超出圖片範圍的區域
    box = Box([90, 90, 120, 120])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (10, 10, 3)


def test_imcropbox_invalid_input():
    # 測試不支援的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        imcropbox(img, "invalid_box")

    # 測試不合法的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        imcropbox(img, np.array([10, 20, 30]))  # 需要4個座標值


def test_imcropbox_padding():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試裁剪超出邊界的情況
    box = Box([10, 20, 150, 120])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (80, 90, 3)

    # 測試裁剪超出邊界並進行填充的情況
    box = Box([10, 20, 150, 120])
    cropped_img = imcropbox(img, box, use_pad=True)
    assert cropped_img.shape == (100, 140, 3)


def test_imcropboxes_custom_boxes():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用自定義Boxes物件進行多個裁剪
    boxes = Boxes([Box([10, 20, 80, 60]), Box([30, 40, 90, 70])])
    cropped_images = imcropboxes(img, boxes)

    assert len(cropped_images) == 2
    for cropped_img in cropped_images:
        assert cropped_img.shape[0] <= 60 - 20
        assert cropped_img.shape[1] <= 80 - 10


def test_imcropboxes_numpy_array():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用NumPy陣列進行多個裁剪
    boxes = np.array([[10, 20, 80, 60], [30, 40, 90, 70]])
    cropped_images = imcropboxes(img, boxes)

    assert len(cropped_images) == 2
    for cropped_img in cropped_images:
        assert cropped_img.shape[0] <= 60 - 20
        assert cropped_img.shape[1] <= 80 - 10


def test_imcropboxes_use_pad():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用填充方式進行多個裁剪
    boxes = np.array([[10, 20, 150, 60], [30, 40, 90, 170]])
    cropped_images = imcropboxes(img, boxes, use_pad=True)

    assert len(cropped_images) == 2
    assert cropped_images[0].shape[0] == 60 - 20
    assert cropped_images[0].shape[1] == 150 - 10
    assert cropped_images[1].shape[0] == 170 - 40
    assert cropped_images[1].shape[1] == 90 - 30


def test_imcropboxes_invalid_input():
    # 測試不支援的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        imcropboxes(img, "invalid_boxes")

    # 測試不合法的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        imcropboxes(img, np.array([[10, 20, 30]]))  # 需要4個座標值


def test_imbinarize_gray_image():
    # 測試用的灰度圖片
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    # 使用THRESH_BINARY進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY)
    assert binarized_img.shape == img.shape
    assert np.unique(binarized_img).tolist() == [0, 255]

    # 使用THRESH_BINARY_INV進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY_INV)
    assert binarized_img.shape == img.shape
    assert np.unique(binarized_img).tolist() == [0, 255]


def test_imbinarize_color_image():
    # 測試用的彩色圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用THRESH_BINARY進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY)
    assert binarized_img.shape == img.shape[:-1]
    assert np.unique(binarized_img).tolist() == [0, 255]

    # 使用THRESH_BINARY_INV進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY_INV)
    assert binarized_img.shape == img.shape[:-1]
    assert np.unique(binarized_img).tolist() == [0, 255]


def test_imbinarize_invalid_input():
    # 測試不支援的圖片維度
    img = np.random.randint(0, 256, size=(100, 100, 100), dtype=np.uint8)
    with pytest.raises(cv2.error):
        imbinarize(img)
