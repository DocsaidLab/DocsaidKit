import pytest

from docsaidkit.torch import calculate_patch_size, list_models_transformers


def test_calculate_patch_size():

    # Test case 1
    image_size = (256, 256)
    num_patches = (4, 4)
    expected_patch_size = (64, 64)
    assert calculate_patch_size(image_size, num_patches) == expected_patch_size

    # Test case 2
    image_size = (512, 512)
    num_patches = (8, 8)
    expected_patch_size = (64, 64)

    assert calculate_patch_size(image_size, num_patches) == expected_patch_size

    # Test case 3 - invalid input
    image_size = (512, 512)
    num_patches = (7, 7)

    with pytest.raises(ValueError):
        calculate_patch_size(image_size, num_patches)
