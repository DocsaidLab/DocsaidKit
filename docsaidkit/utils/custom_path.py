import shutil
from pathlib import Path
from typing import Union

__all__ = ['Path', 'get_curdir', 'rm_path']


def get_curdir(
    path: Union[str, Path],
    absolute: bool = True
) -> Path:
    """
    Function to get the path of current workspace.

    Args:
        path (Union[str, Path]): file path.
        absolute (bool, optional): Whether to return abs path. Defaults to True.

    Returns:
        folder (Union[str, Path]): folder path.
    """
    path = Path(path).absolute() if absolute else Path(path)
    return path.parent.resolve()


def rm_path(path: Union[str, Path]):
    pth = Path(path)
    if pth.is_dir():
        shutil.rmtree(str(path))
    else:
        pth.unlink()
