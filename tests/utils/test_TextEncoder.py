import numpy as np
import pytest
from docsaidkit import TextEncoder


@pytest.fixture
def chars_dict():
    return {
        '[BOS]': 1, '[SEP]': 2, '[EOS]': 3, '[UNK]': 4, '[PAD]': 0,
        'A': 5, 'B': 6, 'C': 7, 'D': 8, '測': 9,
    }


@pytest.fixture
def encoder(chars_dict):
    return TextEncoder(chars_dict=chars_dict, max_length=10)


def test_encoder_initialization(encoder):
    assert encoder.max_length == 10
    assert '[BOS]' in encoder.chars_dict
    assert encoder.special_tokens == (
        '[BOS]', '[SEP]', '[EOS]', '[UNK]', '[PAD]')


def test_encode_single_token(encoder):
    encoded, lengths = encoder.encode(['[BOS]AB'])
    assert np.array_equal(encoded, np.array([[1, 5, 6, 0, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(lengths, np.array([3]))


def test_encode_with_special_tokens(encoder):
    encoded, lengths = encoder.encode(['[BOS]AB[EOS]'])
    assert np.array_equal(encoded, np.array([[1, 5, 6, 3, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(lengths, np.array([4]))


def test_encode_unk_and_max_length(encoder):
    # 此測試用例檢查超過最大長度的字符串是否被截斷，以及未知字符是否正確替換為 [UNK]
    encoded, lengths = encoder.encode(['[BOS]未知字符序列長度超過設定'])
    # 假設 "未知字符序列長度超過設定" 中的所有字符都未在字典中定義，因此將被替換為 [UNK]
    expected_encoded = np.zeros((1, 10), dtype=np.int64)
    expected_encoded[0, 0] = 1  # [BOS]
    expected_encoded[0, 1:] = 4  # [UNK]
    assert np.array_equal(encoded, expected_encoded)
    assert np.array_equal(lengths, np.array([10]))  # 應該是最大長度 10


def test_encode_empty_string(encoder):
    encoded, lengths = encoder.encode([''])
    assert np.array_equal(encoded, np.zeros((1, 10), dtype=np.int64))
    assert np.array_equal(lengths, np.array([0]))
