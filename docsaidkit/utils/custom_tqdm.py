from collections.abc import Sized

from tqdm import tqdm as _tqdm

__all__ = ['Tqdm']


class Tqdm(_tqdm):

    def __init__(self, iterable=None, desc=None, smoothing=0, **kwargs):

        if 'total' in kwargs:
            total = kwargs.pop('total', None)
        else:
            total = len(iterable) if isinstance(iterable, Sized) else None

        super().__init__(
            iterable=iterable,
            desc=desc,
            total=total,
            smoothing=smoothing,
            dynamic_ncols=True,
            **kwargs
        )
