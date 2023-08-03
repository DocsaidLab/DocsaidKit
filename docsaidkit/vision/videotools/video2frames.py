from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

__all__ = ['video2frames']

VIDEO_SUFFIX = ['.MOV', '.MP4', '.AVI', '.WEBM', '.3GP', '.MKV']


def is_video_file(x: Any) -> bool:
    x = Path(x)
    cond1 = x.exists()
    cond2 = x.suffix.upper() in VIDEO_SUFFIX
    return cond1 and cond2


def get_step_inds(start: int, end: int, num: int):
    if num > (end - start):
        raise ValueError(f'num is larger than the number of total frames.')
    return np.around(np.linspace(
        start=start,
        stop=end,
        num=num,
        endpoint=False
    )).astype(int).tolist()


def _extract_frames(
    video_path: str,
    frame_inds: List[int]
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inds[0])

    frames = []
    for _ in frame_inds:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def video2frames(
    video_path: str,
    frame_per_sec: int = None,
    start_sec: float = 0,
    end_sec: float = None,
    n_threads: int = 8,
) -> List[np.ndarray]:
    """
    Extracts the frames from a video using ray
    Inputs:
        video_path (str):
            path to the video.
        frame_per_sec (int, Optional):
            the number of extracting frames per sec.
            If None, all frames will be extracted.
        start_sec (int):
            the start second for frame extraction.
        end_sec (int):
            the end second for frame extraction.
        n_threads (int):
            the number of threads.

    Return:
        frames (list)
            [frame1, frame2, None, ...] or [frame1, frame2, ...,] else []
    """
    if not is_video_file(video_path):
        raise TypeError(f'The video_path {video_path} is inappropriate.')

    # get total_frames frames of video
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if total_frames == 0 or fps == 0:
        return []

    frame_per_sec = fps if frame_per_sec is None else frame_per_sec
    total_sec = total_frames / fps

    # get frame inds
    end_sec = total_sec if end_sec is None or end_sec > total_sec else end_sec
    if start_sec > end_sec:
        raise ValueError(f'The start_sec should less than end_sec. {end_sec}')

    total_sec = end_sec - start_sec
    start_frame = round(start_sec * fps)
    end_frame = round(end_sec * fps)
    num = round(total_sec * frame_per_sec)
    frame_inds = get_step_inds(start_frame, end_frame, num)

    out_frames = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        divide_size = (len(frame_inds) + n_threads - 1) // n_threads
        frame_inds_list = [
            frame_inds[i * divide_size: (i + 1) * divide_size]
            for i in range(n_threads)
        ]

        future_to_frames = {
            executor.submit(_extract_frames, video_path, inds): inds
            for inds in frame_inds_list if len(inds) > 0
        }

        for future in as_completed(future_to_frames):
            frames = future.result()
            out_frames.extend(frames)

    return out_frames
