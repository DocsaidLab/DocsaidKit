from typing import List, Tuple, Union

import cv2
import numpy as np

from ..enums import BORDER, INTER, ROTATE
from ..structures import Polygon, Polygons, order_points_clockwise

__all__ = [
    'imresize', 'imrotate90', 'imrotate', 'imwarp_quadrangle',
    'imwarp_quadrangles'
]


def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    interpolation: Union[str, int, INTER] = INTER.BILINEAR,
    return_scale: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """
    This function is used to resize image.

    Args:
        img (np.ndarray):
            A numpy image.
        size (Tuple[int, int]):
            The size of the resized image. If only one dimension is given,
            calculate the other one maintaining the aspect ratio.
        interpolation (Union[str, int, INTER]):
            Method of interpolation. Default: INTER.BILINEAR.
        return_scale (bool):
            Return scale or not. Default: False.

    Returns:
        img : only resized img or resized img and scale.
    """

    interpolation = INTER.obj_to_enum(interpolation)

    raw_h, raw_w = img.shape[:2]
    h, w = size

    # If only one dimension is given, calculate the other one maintaining
    # the aspect ratio.
    if h is None and w is not None:
        scale = w / raw_w
        h = int(raw_h * scale + 0.5)  # round to nearest integer
    elif w is None and h is not None:
        scale = h / raw_h
        w = int(raw_w * scale + 0.5)  # round to nearest integer

    resized_img = cv2.resize(img, (w, h), interpolation=interpolation.value)

    if return_scale:
        if 'scale' not in locals():  # calculate scale if not already done
            w_scale = w / raw_w
            h_scale = h / raw_h
        else:
            w_scale = h_scale = scale
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def imrotate90(img, rotate_code: ROTATE) -> np.ndarray:
    """
    Function to rotate image with 90-based rotate_code.

    Args:
        img (np.ndarray): Numpy image.
        rotate_code (RotateCode): Rotation code.

    Returns:
        Rotated img (np.ndarray)
    """
    return cv2.rotate(img.copy(), rotate_code)


def imrotate(
    img: np.ndarray,
    angle: float,
    scale: float = 1,
    interpolation: Union[str, int, INTER] = INTER.BILINEAR,
    bordertype: Union[str, int, BORDER] = BORDER.CONSTANT,
    bordervalue: Union[int, Tuple[int, int, int]] = None,
    expand: bool = True,
    center: Tuple[int, int] = None,
) -> np.ndarray:
    '''
    Rotate the image by angle.

    Args:
        img (np.ndarray): Image to be rotated.
        angle (float): In degrees clockwise order.
        interpolation (Union[str, int, Interpolation], optional):
            interpolation type, only works as bordertype is not in constant mode.
            Default to Interpolation.BILINEAR.
        bordertype (BorderType, optional):
            border type. Default to BorderType.CONSTANT.
        bordervalue (Union[int, Tuple[int, int, int]], optional):
            border's filling value, only works as bordertype is BorderType.CONSTANT.
            Defaults to None.
        expand (bool, optional):
            Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
            Defaults to False.
        center (Union[Tuple[int], List[int]], optional):
            Optional center of rotation.
            Default is the center of the image (None).

    Returns:
        rotated img: rotated img.
    '''
    bordertype = BORDER.obj_to_enum(bordertype)
    bordervalue = (bordervalue,) * \
        3 if isinstance(bordervalue, int) else bordervalue
    interpolation = INTER.obj_to_enum(interpolation)

    h, w = img.shape[:2]
    center = center or (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    if expand:
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos)) + 1
        nH = int((h * cos) + (w * sin)) + 1

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        # perform the actual rotation and return the image
        dst = cv2.warpAffine(
            img, M, (nW, nH),
            flags=interpolation,
            borderMode=bordertype,
            borderValue=bordervalue
        )
    else:
        dst = cv2.warpAffine(
            img, M, (w, h),
            flags=interpolation,
            borderMode=bordertype,
            borderValue=bordervalue
        )

    return dst.astype('uint8')


def imwarp_quadrangle(
    img: np.ndarray,
    polygon: Union[Polygon, np.ndarray],
) -> np.ndarray:
    """
    Apply a 4-point perspective transform to an image using a given polygon.

    Args:
        img (np.ndarray):
            The input image to be transformed.
        polygon (Union[Polygon, np.ndarray]):
            The polygon object containing the four points defining the transform.

    Raises:
        TypeError:
            If img is not a numpy ndarray or polygon is not a Polygon object.
        ValueError:
            If the polygon does not contain exactly four points.

    Returns:
        np.ndarray: The transformed image.
    """
    if isinstance(polygon, np.ndarray):
        polygon = Polygon(polygon)

    if not isinstance(polygon, Polygon):
        raise TypeError(
            f'Input type of polygon {type(polygon)} not supported.')

    if len(polygon) != 4:
        raise ValueError(
            f'Input polygon, which is not contain 4 points is invalid.')

    width, height = polygon.min_box_wh
    if width < height:
        width, height = height, width

    src_pts = order_points_clockwise(polygon.numpy())
    w_rect, h_rect = polygon.boundingbox[2:]
    if h_rect / w_rect > 4:
        src_pts = np.roll(src_pts, -1, axis=0)

    dst_pts = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (int(width), int(height)))


def imwarp_quadrangles(
    img: np.ndarray,
    polygons: Polygons,
) -> List[np.ndarray]:
    """
    Apply a 4-point perspective transform to an image using a given polygons.

    Args:
        img (np.ndarray):
            The input image to be transformed.
        polygons (Polygons):
            The polygons object containing the four points defining the transform.

    Returns:
        List[np.ndarray]: The transformed image.
    """
    if not isinstance(polygons, Polygons):
        raise TypeError(
            f'Input type of polygons {type(polygons)} not supported.')
    return [imwarp_quadrangle(img, poly) for poly in polygons]
