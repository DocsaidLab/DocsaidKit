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
    frames = video2frames(video_path, frame_per_sec=2, start_sec=2, end_sec=4)
    assert len(frames) == 4

def test_video2frames_with_start_end_sec():
    # 測試指定提取幀的時間範圍
    frames = video2frames(video_path, start_sec=2, end_sec=4)
    assert len(frames) == 60

def test_video2frames_invalid_input():
    # 測試不支援的影片類型
    with pytest.raises(TypeError):
        video2frames("invalid_video.txt")

    # 測試start_sec大於end_sec
    with pytest.raises(ValueError):
        video2frames(video_path, start_sec=5, end_sec=3)

    # 測試不存在的影片路徑
    with pytest.raises(TypeError):
        video2frames("non_existent_video.mp4")
