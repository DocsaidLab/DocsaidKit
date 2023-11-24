from opencc import OpenCC

__all__ = [
    'OpenCC',
    'convert_simplified_to_traditional',
    'convert_traditional_to_simplified',
]


def convert_simplified_to_traditional(text):
    converter = OpenCC('s2t')  # 簡體中文 -> 繁体中文
    return converter.convert(text)


def convert_traditional_to_simplified(text):
    converter = OpenCC('t2s')  # 繁体中文 -> 簡體中文
    return converter.convert(text)
