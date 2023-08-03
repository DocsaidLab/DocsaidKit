from typing import Tuple, Union

from huggingface_hub import list_models

__all__ = ['list_models_transformers', 'calculate_patch_size']


def list_models_transformers(*args, **kwargs):
    models = list(iter(list_models(*args, **kwargs)))
    return [m.modelId for m in models]


def calculate_patch_size(
    image_size: Union[int, Tuple[int, int]],
    num_patches: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    '''
    Calculate the number of patches that can fit into an image.

    Args:
        image_size (Union[int, Tuple[int, int]]): The size of the image.
        num_patches (Union[int, Tuple[int, int]]): The number of the patch.

    Returns:
        Tuple[int, int]: The number of patches that can fit into the image.
    '''
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if isinstance(num_patches, int):
        num_patches = (num_patches, num_patches)
    if image_size[0] % num_patches[0]:
        raise ValueError(
            f'`image_size` {image_size[0]} can not divided with `{num_patches[0]}`.')
    if image_size[1] % num_patches[1]:
        raise ValueError(
            f'`image_size` {image_size[1]} can not divided with `{num_patches[1]}`.')
    patch_size = (
        image_size[0] // num_patches[0],
        image_size[1] // num_patches[1]
    )
    return patch_size
