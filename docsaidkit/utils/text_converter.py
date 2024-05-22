from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

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
        if self.decode_mode == DecodeMode.CTC:
            masks = (encode != np.roll(encode, 1)) & (encode != 0)
        elif self.decode_mode in [DecodeMode.Default, DecodeMode.Normal]:
            masks = []
            for row in encode:
                eos_index = np.where(row == self.chars_dict["<EOS>"])[0]
                if eos_index.size > 0:
                    mask = np.zeros_like(row, dtype=bool)
                    mask[:eos_index[0]] = True
                else:
                    mask = np.ones_like(row, dtype=bool)
                mask = mask & (row != self.chars_dict["<PAD>"])
                masks.append(mask)

        chars_list = [''.join([self.chars[idx] for idx in e[m]])
                      for e, m in zip(encode, masks)]

        return chars_list

    def __call__(self, *args, **kwargs) -> List[List[str]]:
        return self.decode(*args, **kwargs)


class TextEncoder:

    def __init__(
        self,
        *,
        chars_dict: Dict[str, int],
        max_length: int = 35,
        special_tokens: Tuple[str] = None
    ):
        self.max_length = max_length
        self.chars_dict = chars_dict
        self.chars = {v: k for k, v in self.chars_dict.items()}
        self.special_tokens = special_tokens if special_tokens is not None \
            else ("<BOS>", "<SEP>", "<EOS>", "<UNK>", "<PAD>")

    def encode(self, chars_list: List[Union[str, List[str]]]) -> Tuple[np.ndarray, np.ndarray]:
        encodes = np.zeros((len(chars_list), self.max_length), dtype=np.int64)
        n_strs = np.array([0] * len(chars_list), dtype=np.int32)
        for i, chars in enumerate(chars_list):
            chars_index = []
            pos = 0
            while pos < len(chars):
                matched = False
                # 檢查是否匹配特殊符號
                for token in self.special_tokens:
                    if chars.startswith(token, pos):
                        chars_index.append(self.chars_dict[token])
                        pos += len(token)
                        matched = True
                        break
                # 如果不是特殊符號，處理單個字符
                if not matched:
                    c = chars[pos]
                    if c not in self.chars_dict:
                        c = "<UNK>"
                    chars_index.append(self.chars_dict[c])
                    pos += 1

            n_str = min(len(chars_index), self.max_length)
            encodes[i, :n_str] = chars_index[:n_str]

            # 最後一個字符加上 <EOS>，如果長度超過，則取代最後一個字符
            if encodes[i, n_str - 1] == self.chars_dict["<EOS>"]:
                n_str -= 1
            else:
                if n_str < self.max_length:
                    encodes[i, n_str] = self.chars_dict["<EOS>"]
                else:
                    encodes[i, n_str - 1] = self.chars_dict["<EOS>"]
                    n_str -= 1

            n_strs[i] = n_str

        return encodes, n_strs

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.encode(*args, **kwargs)
