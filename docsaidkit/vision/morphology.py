from typing import Tuple, Union

import cv2
import numpy as np

from ..enums import MORPH

__all__ = [
    'imerode', 'imdilate', 'imopen', 'imclose',
    'imgradient', 'imtophat', 'imblackhat',
]


def imerode(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
) -> np.ndarray:
    """
    Erosion:
        The function erodes the source image using the specified structuring
        element that determines the shape of a pixel neighborhood over which
        the minimum is taken. In case of multi-channel images, each channel is
        processed independently.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.erode(img, cv2.getStructuringElement(kstruct, ksize))


def imdilate(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
) -> np.ndarray:
    """
    Dilation:
        The function dilates the source image using the specified structuring
        element that determines the shape of a pixel neighborhood over which
        the maximum is taken. In case of multi-channel images, each channel is
        processed independently.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.dilate(img, cv2.getStructuringElement(kstruct, ksize))


def imopen(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
) -> np.ndarray:
    """
    Opening:
        The function is another name of erosion followed by dilation.
        It is useful in removing noise.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(kstruct, ksize))


def imclose(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
) -> np.ndarray:
    """
    Closing:
        The function is another name of dilation followed byerosion.
        It is useful in closing small holes inside the foreground objects,
        or small black points on the object.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(kstruct, ksize))


def imgradient(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
) -> np.ndarray:
    """
    Morphological Gradient:
        The function is the difference between dilation and erosion of an image.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, cv2.getStructuringElement(kstruct, ksize))


def imtophat(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
) -> np.ndarray:
    """
    Top Hat:
        The function is the difference between input image and Opening of the image.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(kstruct, ksize))


def imblackhat(
    img: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    kstruct: Union[str, int, MORPH] = MORPH.RECT
):
    """
    Black Hat:
        The function is the difference between input image and Closing of the image.

    Args:
        img (ndarray):
            Input image.
        ksize (Union[int, Tuple[int, int]]):
            Size of the structuring element. Defaults to (3, 3).
        kstruct (MORPH):
            Element shape that could be one of {MORPH.CROSS, MORPH.RECT,
            MORPH.ELLIPSE}, Defaults to MORPH.RECT.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ) * 2
    elif not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError(f'Got inappropriate type or shape of size. {ksize}.')

    kstruct = MORPH.obj_to_enum(kstruct)

    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(kstruct, ksize))
