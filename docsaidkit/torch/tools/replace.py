from typing import Any, Union

import torch.nn as nn

from ..nn import build_nn, build_nn_cls

__all__ = ['has_children', 'replace_module', 'replace_module_attr_value']


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def replace_module(
    model: nn.Module,
    target: Union[type, str],
    dst_module: Union[nn.Module, dict]
) -> None:
    """
    Function to replace modules.

    Args:
        model (nn.Module):
            NN module.
        target (Union[type, str]):
            The type of module you want to replace.
        dst_module (Union[nn.Module, dict]):
            The module you want to use after replacement.
    """
    if not isinstance(dst_module, (nn.Module, dict)):
        raise ValueError(f'dst_module = {dst_module} should be an instance of Module or dict.')

    target = build_nn_cls(target) if isinstance(target, str) else target
    dst_module = build_nn(**dst_module) if isinstance(dst_module, dict) else dst_module

    for name, m in model.named_children():
        if has_children(m):
            replace_module(m, target, dst_module)
        else:
            if isinstance(m, target):
                setattr(model, name, dst_module)


def replace_module_attr_value(
    model: nn.Module,
    target: Union[type, str],
    attr_name: str,
    attr_value: Any
) -> None:
    """
    Function to replace attr's value in target module

    Args:
        model (nn.Module): NN module.
        target (Union[type, str]): The type of module you want to modify.
        attr_name (str): The name of the attribute you want to modify.
        attr_value (Any): The new value of the attribute.
    """
    target = build_nn_cls(target) if isinstance(target, str) else target
    for module in model.modules():
        if isinstance(module, target):
            setattr(module, attr_name, attr_value)
