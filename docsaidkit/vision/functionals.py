from bisect import bisect_left
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ..enums import BORDER
from ..structures import Box, Boxes

__all__ = [
    'meanblur', 'gaussianblur', 'medianblur', 'imcvtcolor', 'imadjust', 'pad',
    'imcropbox', 'imcropboxes', 'imbinarize', 'centercrop',
]

_Ksize = Union[int, Tuple[int, int], np.ndarray]


def _check_ksize(ksize: _Ksize) -> Tuple[int, int]:
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    elif isinstance(ksize, tuple) and len(ksize) == 2 \
            and all(isinstance(val, int) for val in ksize):
        ksize = tuple(ksize)
    elif isinstance(ksize, np.ndarray) and ksize.ndim == 0:
        ksize = (int(ksize), int(ksize))
    else:
        raise TypeError(f'The input ksize = {ksize} is invalid.')

    ksize = tuple(int(val) for val in ksize)
    return ksize


def meanblur(img: np.ndarray, ksize: _Ksize = 3, **kwargs) -> np.ndarray:
    """
    Apply mean blur to the input image.

    Args:
        img (np.ndarray):
            The input image to be blurred.
        ksize (Union[int, Tuple[int, int]], optional):
            The kernel size for blurring.
            If an integer value is provided, a square kernel of the specified
            size will be used. If a tuple (k_height, k_width) is provided, a
            rectangular kernel of the specified size will be used.
            Defaults to 3.

    Returns:
        np.ndarray: The blurred image.
    """
    ksize = _check_ksize(ksize)
    return cv2.blur(img, ksize=ksize, **kwargs)


def gaussianblur(img: np.ndarray, ksize: _Ksize = 3, sigmaX: int = 0, **kwargs) -> np.ndarray:
    """
    Apply Gaussian blur to the input image.

    Args:
        img (np.ndarray):
            The input image to be blurred.
        ksize (Union[int, Tuple[int, int]], optional):
            The kernel size for blurring.
            If an integer value is provided, a square kernel of the specified
            size will be used. If a tuple (k_height, k_width) is provided, a
            rectangular kernel of the specified size will be used.
            Defaults to 3.
        sigmaX (int, optional):
            The standard deviation in the X direction for Gaussian kernel.
            Defaults to 0.

    Returns:
        np.ndarray: The blurred image.
    """
    ksize = _check_ksize(ksize)
    return cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX, **kwargs)


def medianblur(img: np.ndarray, ksize: int = 3, **kwargs) -> np.ndarray:
    """
    Apply median blur to the input image.

    Args:
        img (np.ndarray):
            The input image to be blurred.
        ksize (int, optional):
            The size of the kernel for blurring. It should be an odd integer.
            Defaults to 3.

    Returns:
        np.ndarray: The blurred image.
    """
    ksize = int(ksize)
    return cv2.medianBlur(img, ksize=ksize, **kwargs)


def imcvtcolor(img: np.ndarray, cvt_mode: Union[int, str]) -> np.ndarray:
    """
    Convert the color space of the input image.

    Args:
        img (np.ndarray):
            The input image to be converted.
        cvt_mode (Union[int, str]):
            The color conversion mode. It can be an integer constant representing
            the conversion code or a string representing the OpenCV color
            conversion name.
            For example, 'RGB2GRAY' for converting to grayscale.

    Returns:
        np.ndarray: The image with the desired color space.

    Raises:
        ValueError: If the input cvt_mode is invalid or not supported.
    """
    code = getattr(cv2, f'COLOR_{cvt_mode}', None)
    if code is None:
        raise ValueError(f'Input cvt_mode: "{cvt_mode}" is invaild.')
    img = cv2.cvtColor(img.copy(), code)
    return img


def imadjust(
    img: np.ndarray,
    rng_out: Tuple[int, int] = (0, 255),
    gamma: float = 1.0,
    color_base: str = 'BGR'
) -> np.ndarray:
    """
    Adjust the intensity of an image.

    This function automatically calculates the range of the input image's intensity
    and maps it to a new range specified by rng_out. The intensity values below
    rng_out[0] and above rng_out[1] are clipped.

    Args:
        img (ndarray):
            The input image, should be 2-D or 3-D.
        rng_out (list or tuple, optional):
            The target range of intensity for the output image.
            Defaults to (0, 255).
        gamma (float, optional):
            The value for gamma correction. If gamma is less than 1,
            the mapping is weighted toward higher (brighter) output values.
            If gamma is greater than 1, the mapping is weighted toward lower
            (darker) output values. Defaults to 1.0 (linear mapping).
        color_base (str, optional):
            The color base of the input image. Should be either 'BGR' or 'RGB'.
            Defaults to 'BGR'.

    Returns:
        ndarray: The adjusted image.

    Raises:
        ValueError: If the input image is not 2-D or 3-D.

    Note:
        The histogram is calculated with bin edges [0,256), which results in bins
        with range [0, 255]. The additional 256th bin is not needed for the
        intensity adjustment process.
    """
    is_trans_hsv = False
    if img.ndim == 3:
        img_hsv = imcvtcolor(img, f'{color_base}2HSV')
        v = img_hsv[..., 2]
        is_trans_hsv = True
    else:
        v = img.copy()

    # Stretchlim
    total = v.size
    low_bound, upp_bound = total * 0.01, total * 0.99

    hist, _ = np.histogram(v.ravel(), 256, [0, 256])
    cdf = hist.cumsum()
    rng_in = [bisect_left(cdf, low_bound), bisect_left(cdf, upp_bound)]
    if (rng_in[0] == rng_in[1]) or (rng_in[1] == 0):
        return img

    # Stretching
    dist_in = rng_in[1] - rng_in[0]
    dist_out = rng_out[1] - rng_out[0]

    dst = np.clip((np.clip(v, rng_in[0], None) - rng_in[0]) / dist_in, 0, 1)
    dst = (dst ** gamma) * dist_out + rng_out[0]
    dst = np.clip(dst, rng_out[0], rng_out[1]).astype('uint8')

    if is_trans_hsv:
        img_hsv[..., 2] = dst
        dst = imcvtcolor(img_hsv, f'HSV2{color_base}')

    return dst


def pad(
    img: np.ndarray,
    pad_size: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
    fill_value: Optional[Union[int, Tuple[int, int, int]]] = 0,
    pad_mode: Union[str, int, BORDER] = BORDER.CONSTANT
) -> np.ndarray:
    """
    Pad the input image with specified padding size and mode.

    Args:
        img (np.ndarray):
            The input image to be padded.
        pad_size (Union[int, Tuple[int, int], Tuple[int, int, int, int]]):
            The padding size. It can be an integer to specify equal padding on
            all sides, a tuple (pad_top, pad_bottom, pad_left, pad_right) to
            specify different padding amounts for each side, or a tuple
            (pad_height, pad_width) to specify equal padding for height and width.
        fill_value (Optional[Union[int, Tuple[int, int, int]]], optional):
            The value used for padding. If the input image is a color image (3 channels),
            the fill_value can be an integer or a tuple (R, G, B) to specify the
            color for padding. If the input image is grayscale (1 channel),
            the fill_value should be an integer. Defaults to 0.
        pad_mode (Union[str, int, BORDER], optional):
            The padding mode. Available options are:
                - BORDER.CONSTANT: Pad with constant value (fill_value).
                - BORDER.REPLICATE: Pad by replicating the edge pixels.
                - BORDER.REFLECT: Pad by reflecting the image around the edge.
                - BORDER.REFLECT101: Pad by reflecting the image around the edge,
                    with a slight adjustment to avoid artifacts.
            Defaults to BORDER.CONSTANT.

    Returns:
        np.ndarray: The padded image.
    """
    if isinstance(pad_size, int):
        left = right = top = bottom = pad_size
    elif len(pad_size) == 2:
        top = bottom = pad_size[0]
        left = right = pad_size[1]
    elif len(pad_size) == 4:
        top, bottom, left, right = pad_size
    else:
        raise ValueError(
            f'pad_size is not an int, a tuple with 2 ints, or a tuple with 4 ints.')

    pad_mode = BORDER.obj_to_enum(pad_mode)
    if pad_mode == BORDER.CONSTANT:
        if img.ndim == 3 and isinstance(fill_value, int):
            fill_value = (fill_value, ) * img.shape[-1]
        cond1 = img.ndim == 3 and len(fill_value) == img.shape[-1]
        cond2 = img.ndim == 2 and isinstance(fill_value, int)
        if not (cond1 or cond2):
            raise ValueError(
                f'channel of image is {img.shape[-1]} but length of fill is {len(fill_value)}.')

    img = cv2.copyMakeBorder(
        src=img, top=top, bottom=bottom, left=left, right=right,
        borderType=pad_mode, value=fill_value
    )

    return img


def imcropbox(
    img: np.ndarray,
    box: Union[Box, np.ndarray],
    use_pad: bool = False,
) -> np.ndarray:
    """
    Crop the input image using the provided box.

    This function takes an input image and a cropping box, represented either
    as a custom Box object or as a NumPy array with (x1, y1, x2, y2) coordinates,
    and returns the cropped image.

    Args:
        img (np.ndarray):
            The input image to be cropped.
        box (Union[Box, np.ndarray]):
            The cropping box. It can be either a custom Box object, defined by
            (x1, y1, x2, y2) coordinates, or a NumPy array with the same format.
        use_pad (bool, optional):
            Whether to use padding to handle out-of-boundary regions.
            If set to True, the outside region will be padded with zeros.

    Returns:
        np.ndarray: The cropped image.

    Raises:
        TypeError:
            If the input box is not of type Box or NumPy array.
        ValueError:
            If the input box is not properly defined with (x1, y1, x2, y2)
            coordinates.

    Example:
        # Using custom Box object
        box = Box(100, 100, 300, 200)
        cropped_img = imcropbox(image, box)

        # Using NumPy array
        box = np.array([100, 100, 300, 200])
        cropped_img = imcropbox(image, box)
    """
    if isinstance(box, Box):
        x1, y1, x2, y2 = box.convert("XYXY").numpy().astype(int)
    elif isinstance(box, np.ndarray) and box.ndim == 1 and box.size == 4:
        x1, y1, x2, y2 = box.astype(int)
    else:
        raise TypeError(
            f'Input box is not of type Box or NumPy array with 4 elements.')

    im_h, im_w = img.shape[:2]
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(im_w, x2)
    crop_y2 = min(im_h, y2)
    out = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if use_pad:
        padding = (
            -y1 if y1 < 0 else 0,
            y2 - im_h if y2 > im_h else 0,
            -x1 if x1 < 0 else 0,
            x2 - im_w if x2 > im_w else 0,
        )
        out = pad(out, padding, 0, BORDER.CONSTANT)

    return out


def imcropboxes(
    img: np.ndarray,
    boxes: Union[Boxes, np.ndarray],
    use_pad: bool = False,
) -> List[np.ndarray]:
    """
    Crop the input image using multiple boxes.
    """
    return [imcropbox(img, box, use_pad) for box in boxes]


def imbinarize(
    img: np.ndarray,
    threth: int = cv2.THRESH_BINARY,
    color_base: str = 'BGR'
) -> np.ndarray:
    """
    Function for image binarize.
    Applies a fixed-level threshold to each array element.
    The function is typically used to get a binary image out of a
    grayscale image or for removing a noise, that is, filtering out
    pixels with too small or too large values. There are two types
    of thresholding supported by the function.

    Args:
        img (ndarray):
            Input image. If image with 3 channels, function will apply
            reg2gray function automatically.
        threth (int, optional):
            Threshold type =
            1. cv2.THRESH_BINARY
                    -> cv2.THRESH_OTSU + cv2.THRESH_BINARY
            2. cv2.THRESH_BINARY_INV
                    -> cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV

    Raises:
        ValueError:
            The input image must be 2-D or 3-D.

    Returns:
        dst (ndarray):
            Binary image.
    """
    if img.ndim == 3:
        img = imcvtcolor(img, f'{color_base}2GRAY')
    _, dst = cv2.threshold(img, 0, 255, type=threth + cv2.THRESH_OTSU)
    return dst


def centercrop(img: np.ndarray) -> np.ndarray:
    """
    Crop the input image to a square region centered at the image center.

    Args:
        img (np.ndarray):
            The input image to be cropped.

    Returns:
        np.ndarray: The cropped image.
    """
    box = Box([0, 0, img.shape[1], img.shape[0]]).square()
    return imcropbox(img, box)
