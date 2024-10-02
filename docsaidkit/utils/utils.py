import os
from pprint import pprint
from typing import Any, Generator, Iterable, List, Union

from ..enums import COLORSTR, FORMATSTR

__all__ = [
    'make_batch', 'colorstr', 'gen_download_cmd', 'pprint',
    'download_from_docsaid',
]


def make_batch(
    data: Union[Iterable, Generator],
    batch_size: int
) -> Generator[List, None, None]:
    """
    This function is used to make data to batched data.

    Args:
        generator (Generator): A data generator.
        batch_size (int): batch size of batched data.

    Yields:
        batched data (list): batched data
    """
    batch = []
    for i, d in enumerate(data):
        batch.append(d)
        if (i + 1) % batch_size == 0:
            yield batch
            batch = []
    if batch:
        yield batch


def colorstr(
    obj: Any,
    color: Union[COLORSTR, int, str] = COLORSTR.BLUE,
    fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD
) -> str:
    """
    This function is make colorful string for python.

    Args:
        obj (Any): The object you want to make it print colorful.
        color (Union[COLORSTR, int, str], optional):
            The print color of obj. Defaults to COLORSTR.BLUE.
        fmt (Union[FORMATSTR, int, str], optional):
            The print format of obj. Defaults to FORMATSTR.BOLD.
            Options = {
                'bold', 'underline'
            }

    Returns:
        string: color string.
    """
    if isinstance(color, str):
        color = color.upper()
    if isinstance(fmt, str):
        fmt = fmt.upper()
    color_code = COLORSTR.obj_to_enum(color).value
    format_code = FORMATSTR.obj_to_enum(fmt).value
    color_string = f'\033[{format_code};{color_code}m{obj}\033[0m'
    return color_string


def gen_download_cmd(file_id: str, target: str):
    return f"""
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget
        --quiet
        --save-cookies /tmp/cookies.txt
        --keep-session-cookies
        --no-check-certificate 'https://docs.google.com/uc?export=download&id={file_id}'
        -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={file_id}" -O {target} && rm -rf /tmp/cookies.txt
    """


def download_from_docsaid(file_id: str, file_name: str, target: str):
    os.system(
        f"wget https://cloud.docsaid.org/s/{file_id}/download/{file_name} -O {target}"
    )
