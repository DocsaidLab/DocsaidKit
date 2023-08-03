from threading import Lock, Thread

import cv2
import numpy as np

from ..functionals import imcvtcolor

__all__ = ['IpcamCapture']


class IpcamCapture:

    def __init__(self, url=0, color_base='BGR'):
        """
        Initializes the IpcamCapture class.

        Args:
            url (int, str, optional):
                Identifier for the video source.
                It can be a device index for locally connected cameras or
                a string containing the network address of the IP camera.
                For local cameras, 0 is usually the default camera.
                Defaults to 0.
            color_base (str, optional):
                Color space of the output frames.
                It can be 'BGR' or 'RGB'. Note that the input frames from OpenCV
                are always in BGR format.
                If color_base is set to 'RGB', each frame will be converted from
                BGR to RGB before being returned.
                Defaults to 'BGR'.

        Raises:
            ValueError:
                Raised when the video source cannot be read or does not output
                frames of non-zero dimensions.
        """

        self._capture = cv2.VideoCapture(url)
        self._h = int(self._capture.get(4))
        self._w = int(self._capture.get(3))
        self._frame = None
        self.color_base = color_base.upper()
        self._lock = Lock()

        if self._h == 0 or self._w == 0:
            raise ValueError(f'The image size is not supported.')

        Thread(target=self._queryframe, daemon=True).start()

    def _queryframe(self):
        while 1:
            ret, frame = self._capture.read()
            if not ret:
                break  # Stop the loop if the video stream has ended or is unreadable
            if self.color_base != 'BGR':
                frame = imcvtcolor(frame, cvt_mode=f'BGR2{self.color_base}')
            with self._lock:
                self._frame = frame

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
            else:
                frame = self._frame.copy()
        return frame

    def __iter__(self):
        yield self.get_frame()
