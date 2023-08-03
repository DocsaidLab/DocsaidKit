import torch

from docsaidkit.torch.nn import GradientReversalLayer


def test_gradient_reversal_layer():
    input_ = torch.rand(2, 3, 4, 5, requires_grad=True)
    module = GradientReversalLayer()

    # Test forward pass
    output = module(input_)
    assert output.shape == input_.shape
    assert torch.allclose(output, input_)

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert input_.grad is not None
    assert torch.allclose(input_.grad, torch.tensor(-0.00025))


def test_gradient_reversal_layer_warm_up():
    input_ = torch.rand(2, 3, 4, 5, requires_grad=True)
    module = GradientReversalLayer(warm_up=100)

    # Test forward pass
    output = module(input_)
    assert output.shape == input_.shape
    assert torch.allclose(output, input_)

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert input_.grad is not None
    assert torch.allclose(input_.grad, torch.tensor(-0.01))
