from opencc import OpenCC

__all__ = [
    'OpenCC',
    'convert_simplified_to_traditional',
    'convert_traditional_to_simplified',
]


def convert_simplified_to_traditional(text):
    converter = OpenCC('s2t')  # 从简体中文转换到繁体中文
    return converter.convert(text)


def convert_traditional_to_simplified(text):
    converter = OpenCC('t2s')  # 从繁体中文转换到简体中文
    return converter.convert(text)
