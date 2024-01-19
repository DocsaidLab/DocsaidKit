import errno
import hashlib
import os
from typing import Any, List, Tuple, Union

import dill
import numpy as np
import ujson
import yaml
from natsort import natsorted

from .custom_path import Path
from .custom_tqdm import Tqdm

__all__ = [
    'gen_md5', 'get_files', 'load_json', 'dump_json', 'load_pickle',
    'dump_pickle', 'load_yaml', 'dump_yaml', 'img_to_md5',
]


def gen_md5(file: Union[str, Path], block_size: int = 256 * 128) -> str:
    """
    This function is to gen md5 based on given file.

    Args:
        file (Union[str, Path]): filename
        block_size (int, optional): reading size per loop. Defaults to 256*128.

    Raises:
        ValueError: check file is exist.

    Returns:
        md5 (str)
    """
    with open(str(file), 'rb') as f:
        md5 = hashlib.md5()
        for chunk in iter(lambda: f.read(block_size), b''):
            md5.update(chunk)
    return str(md5.hexdigest())


def img_to_md5(img: np.ndarray) -> str:
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    image_bytes = img.tobytes()
    md5_hash = hashlib.md5(image_bytes)
    return str(md5_hash.hexdigest())


def load_json(path: Union[Path, str], **kwargs) -> dict:
    """
    Function to read ujson.

    Args:
        path (Union[Path, str]): ujson file.

    Returns:
        dict: ujson load to dictionary
    """
    with open(str(path), 'r') as f:
        data = ujson.load(f, **kwargs)
    return data


def dump_json(obj: Any, path: Union[str, Path] = None, **kwargs) -> None:
    """
    Function to write obj to ujson

    Args:
        obj (Any): Object to write to a ujson
        path (Union[str, Path]): ujson file's path
    """
    dump_options = {
        'sort_keys': False,
        'indent': 2,
        'ensure_ascii': False,
        'escape_forward_slashes': False,
    }
    dump_options.update(kwargs)

    if path is None:
        path = Path.cwd() / 'tmp.json'

    with open(str(path), 'w') as f:
        ujson.dump(obj, f, **dump_options)


def get_files(
    folder: Union[str, Path],
    suffix: Union[str, List[str], Tuple[str]] = None,
    recursive: bool = True,
    return_pathlib: bool = True,
    sort_path: bool = True,
    ignore_letter_case: bool = True,
):
    """
    Function to getting all files in the folder with given suffix.

    Args:
        folder (Union[str, Path]):
            An existing folder.
        suffix (Union[str, List[str], Tuple[str]], optional):
            Get all files with same suffix, ex: ['.jpg', '.png'].
            Default is None, which means getting all files in the folder.
        recursive (bool, optional):
            Whether to getting all files includeing subfolders. Defaults to True.
        return_pathlib (bool, optional):
            Whether to return Path or not. Defaults to True.
        sort_path (bool, optional):
            Whether to returning a list of path with nature sorting. Defaults to True.
        ignore_letter_case (bool, optional):
            Whether to getting files with the 'suffix' including lower and upper latter case.
            Defaults to True.

    Raises:
        TypeError:
            check folder is exist.
            suffix must be a list or a string.

    Returns:
        files (list):
            Output a list of files' path in absolute mode.
    """

    # checking folders
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), str(folder))

    if not isinstance(suffix, (str, list, tuple)) and suffix is not None:
        raise TypeError('suffix must be a string, list or tuple.')

    # checking suffix
    suffix = [suffix] if isinstance(suffix, str) else suffix
    if suffix is not None and ignore_letter_case:
        suffix = [s.lower() for s in suffix]

    if recursive:
        files_gen = folder.rglob('*')
    else:
        files_gen = folder.glob('*')

    files = []
    for f in Tqdm(files_gen, leave=False):
        if suffix is None or (ignore_letter_case and f.suffix.lower() in suffix) \
                or (not ignore_letter_case and f.suffix in suffix):
            files.append(f.absolute())

    if not return_pathlib:
        files = [str(f) for f in files]

    if sort_path:
        files = natsorted(files, key=lambda path: str(path).lower())

    return files


def load_pickle(path: Union[str, Path]):
    """
    Function to load a pickle.

    Args:
        path (Union[str, Path]): file path.

    Returns:
        loaded_pickle (dict): loaded pickle.
    """
    with open(str(path), 'rb') as f:
        return dill.load(f)


def dump_pickle(obj, path: Union[str, Path]):
    """
    Function to dump an obj to a pickle file.

    Args:
        obj: object to be dump.
        path (Union[str, Path]): file path.
    """
    with open(str(path), 'wb') as f:
        dill.dump(obj, f)


def load_yaml(path: Union[Path, str]) -> dict:
    """
    Function to read yaml.

    Args:
        path (Union[Path, str]): yaml file.

    Returns:
        dict: yaml load to dictionary
    """
    with open(str(path), 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def dump_yaml(obj, path: Union[str, Path] = None, **kwargs):
    """
    Function to dump an obj to a yaml file.

    Args:
        obj: object to be dump.
        path (Union[str, Path]): file path.
    """
    dump_options = {
        'indent': 2,
        'sort_keys': True
    }
    dump_options.update(kwargs)

    if path is None:
        path = Path.cwd() / 'tmp.yaml'

    with open(str(path), 'w') as f:
        yaml.dump(obj, f, **dump_options)
