from pathlib import Path

import numpy as np
import pytest

from docsaidkit import get_curdir, imread, imwrite

DIR = get_curdir(__file__)


def test_imread():
    # 測試圖片路徑
    image_path = DIR.parent / 'resources' / 'lena.png'

    # 測試 BGR 格式的圖片讀取
    img_bgr = imread(image_path, color_base='BGR')
    assert isinstance(img_bgr, np.ndarray)
    assert img_bgr.shape[-1] == 3  # BGR圖片的channel數為3

    # 測試灰階格式的圖片讀取
    img_gray = imread(image_path, color_base='GRAY')
    assert isinstance(img_gray, np.ndarray)
    assert len(img_gray.shape) == 2  # 灰階圖片的channel數為1

    # 測試不存在的圖片路徑
    with pytest.raises(FileExistsError):
        imread('non_existent_image.jpg')


def test_imwrite():
    # 測試用的圖片
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # 建立一個全黑的BGR圖片

    # 測試BGR格式的圖片寫入
    temp_file_path = DIR / 'temp_image.jpg'
    assert imwrite(img, path=temp_file_path, color_base='BGR')
    assert Path(temp_file_path).exists()

    # 測試不指定路徑時的圖片寫入
    assert imwrite(img, color_base='BGR')  # 將會寫入一個暫時的檔案
