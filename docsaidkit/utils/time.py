import time
from datetime import datetime
from time import struct_time
from typing import Union

import numpy as np

from .utils import colorstr

__all__ = [
    'Timer',
    'now',
    'timestamp2datetime',
    'timestamp2time',
    'timestamp2str',
    'time2datetime',
    'time2timestamp',
    'time2str',
    'datetime2time',
    'datetime2timestamp',
    'datetime2str',
    'str2time',
    'str2datetime',
    'str2timestamp',
]

__doc__ = """
The following directives can be embedded in the format string.
They are shown without the optional field width and precision specification,
and are replaced by the indicated characters in the strftime() result:

==========|==========================================================
Directive | Meaning
==========|==========================================================
    %a    |  Locale’s abbreviated weekday name.
    %A    |  Locale’s full weekday name.
    %b    |  Locale’s abbreviated month name.
    %B    |  Locale’s full month name.
    %c    |  Locale’s appropriate date and time representation.
    %d    |  Day of the month as a decimal number [01,31].
    %H    |  Hour (24-hour clock) as a decimal number [00,23].
    %I    |  Hour (12-hour clock) as a decimal number [01,12].
    %j    |  Day of the year as a decimal number [001,366].
    %m    |  Month as a decimal number [01,12].
    %M    |  Minute as a decimal number [00,59].
    %p    |  Locale’s equivalent of either AM or PM.
    %S    |  Second as a decimal number [00,61].
    %U    |  Week number of the year (Sunday as the first day of the week)
            |  as a decimal number [00,53]. All days in a new year preceding
            |  the first Sunday are considered to be in week 0.
    %w    |  Weekday as a decimal number [0(Sunday),6].
    %W    |  Week number of the year (Monday as the first day of the week)
            |  as a decimal number [00,53]. All days in a new year preceding
            |  the first Monday are considered to be in week 0.
    %x    |  Locale’s appropriate date representation.
    %X    |  Locale’s appropriate time representation.
    %y    |  Year without century as a decimal number [00,99].
    %Y    |  Year with century as a decimal number.
    %z    |  Time zone offset indicating a positive or negative time difference
            |  from UTC/GMT of the form +HHMM or -HHMM, where H represents decimal
            |  hour digits and M represents decimal minute digits [-23:59, +23:59].
    %Z    |  Time zone name (no characters if no time zone exists).
    %%    |  A literal '%' character.
==========|==========================================================

Notes:
    1.  When used with the strptime() function, the %p directive only affects the
        output hour field if the %I directive is used to parse the hour.
    2.  The range really is 0 to 61; value 60 is valid in timestamps representing
        leap seconds and value 61 is supported for historical reasons.
    3.  When used with the strptime() function, %U and %W are only used in calculations
        when the day of the week and the year are specified.
"""


class Timer:
    """
    Usage:
        1. Using 'tic' and 'toc' method:
            t = Timer()

            t.tic()
            do something...
            t.toc()

        2. Using decorator:

            @ Timer()
            def testing_function(*args, **kwargs):
                do something...

        3. Using 'with' statement

            with Timer():
                do something...
    """

    def __init__(self, precision: int = 5, desc: str = None, verbose: bool = False):
        self.precision = precision
        self.desc = desc
        self.verbose = verbose
        self.__record = []

    def tic(self):
        """ start timer """
        if self.desc is not None and self.verbose:
            print(colorstr(self.desc, 'yellow'))
        self.time = time.perf_counter()

    def toc(self, verbose=False):
        """ get time lag from start """
        if getattr(self, 'time', None) is None:
            raise ValueError(
                f'The timer has not been started. Tic the timer first.')
        total = round(time.perf_counter() - self.time, self.precision)

        if verbose or self.verbose:
            print(colorstr(f'Cost: {total} sec', 'white'))

        self.__record.append(total)
        return total

    def __call__(self, fcn):
        def warp(*args, **kwargs):
            self.tic()
            result = fcn(*args, **kwargs)
            self.toc()
            return result
        return warp

    def __enter__(self):
        self.tic()

    def __exit__(self, type, value, traceback):
        self.dt = self.toc(True)

    def clear_record(self):
        self.__record = []

    @ property
    def mean(self):
        if len(self.__record):
            return np.array(self.__record).mean().round(self.precision)

    @ property
    def max(self):
        if len(self.__record):
            return np.array(self.__record).max().round(self.precision)

    @ property
    def min(self):
        if len(self.__record):
            return np.array(self.__record).min().round(self.precision)

    @ property
    def std(self):
        if len(self.__record):
            return np.array(self.__record).std().round(self.precision)


def now(typ: str = 'timestamp', fmt: str = None):
    """
    Get now time. Specify the output type of time, or give the
    formatted rule to get the time string, eg: now(fmt='%Y-%m-%d').

    Args:
        typ (str, optional):
            Return now time with specific type.
            Supporting type:{'timestamp', 'datetime', 'time'}, Defaults to 'timestamp'.

    Raises:
        ValueError: Unsupported type error.
    """
    if typ == 'timestamp':
        t = time.time()
    elif typ == 'datetime':
        t = datetime.now()
    elif typ == 'time':
        t = time.gmtime(time.time())
    else:
        raise ValueError(f'Unsupported input {typ} type of time.')

    if fmt != None:
        t = timestamp2str(time.time(), fmt=fmt)

    return t


def timestamp2datetime(ts: Union[int, float]):
    return datetime.fromtimestamp(ts)


def timestamp2time(ts: Union[int, float]):
    return time.localtime(ts)


def timestamp2str(ts: Union[int, float], fmt: str):
    return time2str(timestamp2time(ts), fmt)


def time2datetime(t: struct_time):
    if not isinstance(t, struct_time):
        raise TypeError(f'Input type: {type(t)} error.')
    return datetime(*t[0:6])


def time2timestamp(t: struct_time):
    if not isinstance(t, struct_time):
        raise TypeError(f'Input type: {type(t)} error.')
    return time.mktime(t)


def time2str(t: struct_time, fmt: str):
    if not isinstance(t, struct_time):
        raise TypeError(f'Input type: {type(t)} error.')
    return time.strftime(fmt, t)


def datetime2time(dt: datetime):
    if not isinstance(dt, datetime):
        raise TypeError(f'Input type: {type(dt)} error.')
    return dt.timetuple()


def datetime2timestamp(dt: datetime):
    if not isinstance(dt, datetime):
        raise TypeError(f'Input type: {type(dt)} error.')
    return dt.timestamp()


def datetime2str(dt: datetime, fmt: str):
    if not isinstance(dt, datetime):
        raise TypeError(f'Input type: {type(dt)} error.')
    return dt.strftime(fmt)


def str2time(s: str, fmt: str):
    if not isinstance(s, str):
        raise TypeError(f'Input type: {type(s)} error.')
    return time.strptime(s, fmt)


def str2datetime(s: str, fmt: str):
    if not isinstance(s, str):
        raise TypeError(f'Input type: {type(s)} error.')
    return datetime.strptime(s, fmt)


def str2timestamp(s: str, fmt: str):
    if not isinstance(s, str):
        raise TypeError(f'Input type: {type(s)} error.')
    return time2timestamp(str2time(s, fmt))
