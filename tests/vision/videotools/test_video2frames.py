import numpy as np
import pytest
from docsaidkit import get_curdir, video2frames

# 測試用的影片
video_path = get_curdir(__file__) / "video_test.mp4"


def test_video2frames():
    # 測試從影片中提取所有幀
    frames = video2frames(video_path)
    assert isinstance(frames, list)
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)


def test_video2frames_with_fps():
    # 測試指定提取幀的速度
    frames = video2frames(video_path, frame_per_sec=2)
    assert len(frames) == 18


def test_video2frames_invalid_input():
    # 測試不支援的影片類型
    with pytest.raises(TypeError):
        video2frames("invalid_video.txt")

    # 測試不存在的影片路徑
    with pytest.raises(TypeError):
        video2frames("non_existent_video.mp4")
