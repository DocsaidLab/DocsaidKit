from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from .components import build_activation

__all__ = [
    'PowerModule', 'initialize_weights', 'WeightedSum', 'Identity',
    'Transpose', 'Permute',
]


def initialize_weights(
    model: nn.Module,
    init_type: str = 'normal',
    recursive: bool = True
) -> None:
    """
    Initialize the weights in the given model.

    Args:
        model (nn.Module):
            The model to initialize.
        init_type (str, optional):
            The initialization method to use. Supported options are 'uniform'
            and 'normal'. Defaults to 'normal'.
        recursive (bool, optional):
            Whether to recursively initialize child modules. Defaults to True.

    Raises:
        TypeError: If init_type is not supported.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f'model must be an instance of nn.Module, but got {type(model)}')

    init_functions = {
        'uniform': nn.init.kaiming_uniform_,
        'normal': nn.init.kaiming_normal_
    }

    if init_type not in init_functions:
        raise TypeError(f'init_type {init_type} is not supported.')
    nn_init = init_functions[init_type]

    def _recursive_init(m):
        for child in m.children():
            if len(list(child.children())) > 0 and recursive:
                _recursive_init(child)
            else:
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    nn_init(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                    if child.affine:
                        nn.init.ones_(child.weight)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)

    _recursive_init(model)


class PowerModule(nn.Module):
    """
    A module that provides additional functionality for weight initialization,
    freezing and melting layers.
    """

    def initialize_weights_(self, init_type: str = 'normal') -> None:
        """
        Initialize the weights of the module.

        Args:
            init_type (str): The type of initialization. Can be 'normal' or 'uniform'.
        """
        initialize_weights(self, init_type)

    def freeze(self, part_names: Union[str, List[str]] = 'all', verbose: bool = False) -> None:
        """
        Freeze the parameters of specified layers.

        Args:
            part_names (Union[str, List[str]]): The names of the layers to freeze.
                If 'all', all layers are frozen.
            verbose (bool): Whether to print messages indicating which layers were frozen.
        """
        if part_names == 'all':
            for name, params in self.named_parameters():
                if verbose:
                    print(f'Freezing layer {name}')
                params.requires_grad_(False)
        elif part_names is None:
            return
        else:
            part_names = [part_names] if isinstance(part_names, str) \
                else part_names
            for layer_name in part_names:
                module = self
                for attr in layer_name.split('.'):
                    module = getattr(module, attr)
                for name, param in module.named_parameters():
                    if verbose:
                        print(f'Freezing layer {layer_name}.{name}')
                    param.requires_grad_(False)

    def melt(self, part_names: Union[str, List[str]] = 'all', verbose: bool = False) -> None:
        """
        Unfreeze the parameters of specified layers.

        Args:
            part_names (Union[str, List[str]]): The names of the layers to unfreeze.
                If 'all', all layers are unfrozen.
            verbose (bool): Whether to print messages indicating which layers were unfrozen.
        """
        if part_names == 'all':
            for name, params in self.named_parameters():
                if verbose:
                    print(f'Unfreezing layer {name}')
                params.requires_grad_(True)
        elif part_names is None:
            return
        else:
            part_names = [part_names] if isinstance(part_names, str) \
                else part_names
            for layer_name in part_names:
                module = self
                for attr in layer_name.split('.'):
                    module = getattr(module, attr)
                for name, param in module.named_parameters():
                    if verbose:
                        print(f'Unfreezing layer {layer_name}.{name}')
                    param.requires_grad_(True)


class WeightedSum(nn.Module):

    def __init__(
        self,
        input_size: int,
        act: Optional[Union[dict, nn.Module]] = None,
        requires_grad: bool = True,
    ) -> None:
        """
        Initializes a WeightedSum module.

        Args:
            input_size (int):
                The number of inputs to be summed.
            act Optional[Union[dict, nn.Module]]:
                Optional activation function or dictionary of its parameters.
                Defaults to None.
            requires_grad (bool, optional):
                Whether to require gradients for the weights. Defaults to True.
        """
        super().__init__()
        self.input_size = input_size
        self.weights = nn.Parameter(
            torch.ones(input_size, dtype=torch.float32),
            requires_grad=requires_grad
        )
        self.weights_relu = nn.ReLU()
        if act is None:
            self.relu = nn.Identity()
        else:
            self.relu = act if isinstance(act, nn.Module) \
                else build_activation(**act)
        self.epsilon = 1e-4

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != self.input_size:
            raise ValueError('Invalid input size not equal to weight size.')
        weights = self.weights_relu(self.weights)
        weights = weights / (
            torch.sum(weights, dim=0, keepdim=True) + self.epsilon)
        weighted_x = torch.einsum(
            'i,i...->...', weights, torch.stack(x, dim=0))
        weighted_x = self.relu(weighted_x)
        return weighted_x


class Identity(PowerModule):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class Transpose(nn.Module):

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


class Permute(nn.Module):

    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)
