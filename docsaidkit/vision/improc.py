import warnings
from pathlib import Path
from typing import Any, List, Union

import cv2
import numpy as np
import piexif
import pybase64
import pyheif
from pdf2image import convert_from_bytes, convert_from_path
from turbojpeg import TurboJPEG

from ..enums import IMGTYP, ROTATE
from .functionals import imcvtcolor
from .geometric import imrotate90

__all__ = [
    'imread', 'imwrite', 'imencode', 'imdecode', 'img_to_b64', 'img_to_b64str',
    'b64_to_img', 'b64str_to_img', 'b64_to_npy', 'b64str_to_npy', 'npy_to_b64',
    'npy_to_b64str', 'npyread', 'pdf2imgs', 'jpgencode', 'jpgdecode', 'jpgread',
    'pngencode', 'pngdecode', 'is_numpy_img', 'get_orientation_code'
]

jpeg = TurboJPEG()


def is_numpy_img(x: Any) -> bool:
    """
    x == ndarray (H x W x C)
    """
    return isinstance(x, np.ndarray) and (x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in [1, 3]))


def get_orientation_code(stream: Union[str, Path, bytes]):
    code = None
    try:
        exif_dict = piexif.load(stream)
        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
            if orientation == 3:
                code = ROTATE.ROTATE_180
            elif orientation == 6:
                code = ROTATE.ROTATE_90
            elif orientation == 8:
                code = ROTATE.ROTATE_270
    finally:
        return code


def jpgencode(img: np.ndarray, quality: int = 90) -> Union[bytes, None]:
    byte_ = None
    if is_numpy_img(img):
        try:
            byte_ = jpeg.encode(img, quality=quality)
        except:
            pass
    return byte_


def jpgdecode(byte_: bytes) -> Union[np.ndarray, None]:
    try:
        bgr_array = jpeg.decode(byte_)
        code = get_orientation_code(byte_)
        bgr_array = imrotate90(
            bgr_array, code) if code is not None else bgr_array
    except:
        bgr_array = None

    return bgr_array


def jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]:
    with open(str(img_file), 'rb') as f:
        binary_img = f.read()
        bgr_array = jpgdecode(binary_img)

    return bgr_array


def pngencode(img: np.ndarray, compression: int = 1) -> Union[bytes, None]:
    byte_ = None
    if is_numpy_img(img):
        try:
            byte_ = cv2.imencode('.png', img, params=[int(
                cv2.IMWRITE_PNG_COMPRESSION), compression])[1].tobytes()
        except:
            pass
    return byte_


def pngdecode(byte_: bytes) -> Union[np.ndarray, None]:
    try:
        enc = np.frombuffer(byte_, 'uint8')
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    except:
        img = None
    return img


def imencode(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]:
    IMGTYP = IMGTYP.obj_to_enum(IMGTYP)
    encode_fn = jpgencode if IMGTYP == IMGTYP.JPEG else pngencode
    byte_ = encode_fn(img)
    return byte_


def imdecode(byte_: bytes) -> Union[np.ndarray, None]:
    try:
        img = jpgdecode(byte_)
        img = pngdecode(byte_) if img is None else img
    except:
        img = None
    return img


def img_to_b64(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]:
    IMGTYP = IMGTYP.obj_to_enum(IMGTYP)
    encode_fn = jpgencode if IMGTYP == IMGTYP.JPEG else pngencode
    try:
        b64 = pybase64.b64encode(encode_fn(img))
    except:
        b64 = None
    return b64


def npy_to_b64(x: np.ndarray, dtype='float32') -> bytes:
    return pybase64.b64encode(x.astype(dtype).tobytes())


def npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str:
    return pybase64.b64encode(x.astype(dtype).tobytes()).decode(string_encode)


def img_to_b64str(
    img: np.ndarray,
    IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG,
    string_encode: str = 'utf-8'
) -> Union[str, None]:
    b64 = img_to_b64(img, IMGTYP)
    return b64.decode(string_encode) if isinstance(b64, bytes) else None


def b64_to_img(b64: bytes) -> Union[np.ndarray, None]:
    try:
        img = imdecode(pybase64.b64decode(b64))
    except:
        img = None
    return img


def b64str_to_img(
    b64str: Union[str, None],
    string_encode: str = 'utf-8'
) -> Union[np.ndarray, None]:

    if b64str is None:
        warnings.warn("b64str is None.")
        return None

    if not isinstance(b64str, str):
        raise ValueError("b64str is not a string.")

    return b64_to_img(b64str.encode(string_encode))


def b64_to_npy(x: bytes, dtype='float32') -> np.ndarray:
    return np.frombuffer(pybase64.b64decode(x), dtype=dtype)


def b64str_to_npy(x: bytes, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray:
    return np.frombuffer(pybase64.b64decode(x.encode(string_encode)), dtype=dtype)


def npyread(path: Union[str, Path]) -> Union[np.ndarray, None]:
    try:
        with open(str(path), 'rb') as f:
            img = np.load(f)
    except:
        img = None
    return img


def read_heic_to_numpy(file_path: str):
    heif_file = pyheif.read(file_path)
    data = heif_file.data
    if heif_file.mode == "RGB":
        numpy_array = np.frombuffer(data, dtype=np.uint8).reshape(
            heif_file.size[1], heif_file.size[0], 3)
    elif heif_file.mode == "RGBA":
        numpy_array = np.frombuffer(data, dtype=np.uint8).reshape(
            heif_file.size[1], heif_file.size[0], 4)
    else:
        raise ValueError("Unsupported HEIC color mode")
    return numpy_array


def imread(
    path: Union[str, Path],
    color_base: str = 'BGR',
    verbose: bool = False
) -> Union[np.ndarray, None]:
    """
    This function reads an image from a given file path and converts its color
    base if necessary.

    Args:
        path (Union[str, Path]):
            The path to the image file to be read.
        color_base (str, optional):
            The desired color base for the image. If not 'BGR', will attempt to
            convert using 'imcvtcolor' function. Defaults to 'BGR'.
        verbose (bool, optional):
            If set to True, a warning will be issued when the read image is None.
            Defaults to False.

    Raises:
        FileExistsError:
            If the image file at the specified path does not exist.

    Returns:
        Union[np.ndarray, None]:
            The image as a numpy ndarray if successful, None otherwise.
    """
    if not Path(path).exists():
        raise FileExistsError(f'{path} can not found.')

    if Path(path).suffix.lower() == '.heic':
        img = read_heic_to_numpy(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = jpgread(path)
        img = cv2.imread(str(path)) if img is None else img

    if img is None:
        if verbose:
            warnings.warn("Got a None type image.")
        return

    if color_base != 'BGR':
        img = imcvtcolor(img, cvt_mode=f'BGR2{color_base}')

    return img


def imwrite(
    img: np.ndarray,
    path: Union[str, Path] = None,
    color_base: str = 'BGR',
    suffix: str = '.jpg',
) -> bool:
    """
    Writes an image to a file with optional color base conversion.

    Args:
        img (np.ndarray):
            The image to write, as a numpy ndarray.
        path (Union[str, Path], optional):
            The path where to write the image file. If None, writes to a temporary
            file. Defaults to None.
        color_base (str, optional):
            The current color base of the image. If not 'BGR', the function will
            attempt to convert it to 'BGR'. Defaults to 'BGR'.
        suffix (str, optional):
            The suffix of the temporary file if path is None. Defaults to '.jpg'.

    Returns:
        bool: True if the write operation is successful, False otherwise.
    """
    color_base = color_base.upper()
    if color_base != 'BGR':
        img = imcvtcolor(img, cvt_mode=f'{color_base}2BGR')
    return cv2.imwrite(str(path) if path else f'tmp{suffix}', img)


def pdf2imgs(stream: Union[str, Path, bytes]) -> Union[List[np.ndarray], None]:
    """
    Function for converting a PDF document to numpy images.

    Args:
        file_dir (str): A path of PDF document.

    Returns:
        img: Images will be a list of np image representing each page of the PDF document.
    """
    try:
        if isinstance(stream, bytes):
            pil_imgs = convert_from_bytes(stream)
        else:
            pil_imgs = convert_from_path(stream)
        return [imcvtcolor(np.array(img), cvt_mode='RGB2BGR') for img in pil_imgs]
    except:
        return
