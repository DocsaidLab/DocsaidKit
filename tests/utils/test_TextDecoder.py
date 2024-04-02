import numpy as np
import pytest
from docsaidkit import DecodeMode, TextDecoder


@pytest.fixture
def chars_dict():
    return {
        '[BOS]': 1, '[SEP]': 2, '[EOS]': 3, '[UNK]': 4, '[PAD]': 0,
        'A': 5, 'B': 6, 'C': 7, 'D': 8, '測': 9,
    }


@pytest.fixture
def decoder(chars_dict):
    return TextDecoder(chars_dict=chars_dict, decode_mode=DecodeMode.Default)


def test_decoder_initialization(decoder):
    assert decoder.decode_mode == DecodeMode.Default
    assert '[BOS]' in decoder.chars_dict


def test_decode_basic(decoder):
    # 假設編碼後的數據是 `[1, 5, 6, 0, 0]`，對應於 '[BOS]嶲⒣[PAD][PAD]'
    encoded_data = np.array([[1, 5, 6, 0, 0]])
    decoded = decoder.decode(encoded_data)
    # 由於在 Default 和 CTC 模式下，[PAD] 被忽略，期望輸出為除去 [PAD] 的解碼結果
    assert decoded == ['[BOS]AB']


def test_decode_with_mode_ctc(decoder):
    # 這個測試設定解碼模式為 CTC
    decoder.decode_mode = DecodeMode.CTC
    encoded_data = np.array([[1, 1, 5, 0, 6, 6, 0]])
    decoded = decoder.decode(encoded_data)
    # 在 CTC 模式下，重複的字符會被合併，[PAD] 被忽略
    assert decoded == ['[BOS]AB']


def test_decode_with_mode_normal(decoder):
    # 這個測試設定解碼模式為 Normal
    decoder.decode_mode = DecodeMode.Normal
    encoded_data = np.array([[1, 0, 5, 6, 0]])
    decoded = decoder.decode(encoded_data)
    # 在 Normal 模式下，不會合併重複的字符，但 [PAD] 仍被忽略
    assert decoded == ['[BOS]AB']
