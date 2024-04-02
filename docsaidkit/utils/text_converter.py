from enum import IntEnum
from typing import Dict, List, Optional, Union

import numpy as np

from ..mixins import EnumCheckMixin

__all__ = ['DecodeMode', 'TextDecoder', 'TextEncoder']


class DecodeMode(EnumCheckMixin, IntEnum):
    Default = 0
    CTC = 1
    Normal = 2


class TextDecoder:

    def __init__(
        self,
        *,
        chars_dict: Dict[str, int],
        decode_mode: Optional[Union[DecodeMode, str, int]] = DecodeMode.Default
    ):
        self.chars_dict = chars_dict
        self.chars = {v: k for k, v in self.chars_dict.items()}
        self.decode_mode = DecodeMode.obj_to_enum(decode_mode)

    def decode(self, encode: List[np.ndarray]) -> List[List[str]]:
        encode = np.array(encode)
        if self.decode_mode in [DecodeMode.Default, DecodeMode.CTC]:
            masks = (encode != np.roll(encode, 1)) & (encode != 0)
        elif self.decode_mode == DecodeMode.Normal:
            masks = encode != 0
        chars_list = [''.join([self.chars[idx] for idx in e[m]])
                      for e, m in zip(encode, masks)]

        return chars_list


class TextEncoder:

    def __init__(
        self,
        *,
        chars_dict: Dict[str, int],
        max_length: int = 35
    ):
        self.chars_dict = chars_dict
        self.chars = {v: k for k, v in self.chars_dict.items()}
        self.max_length = max_length

    def encode(self, chars_list: List[Union[str, List[str]]]):
        encodes = np.zeros((len(chars_list), self.max_length), dtype=np.int64)
        n_strs = np.array([0] * len(chars_list), dtype=np.int32)
        for i, chars in enumerate(chars_list):
            chars_index = []
            for c in chars:
                if c not in self.chars_dict:
                    c = '[UNK]'
                chars_index.append(self.chars_dict[c])

            n_str = min(len(chars), self.max_length)
            encodes[i, :n_str] = chars_index[:n_str]
            n_strs[i] = n_str

        return encodes, n_strs
