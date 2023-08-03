import pytest
import torch

from docsaidkit.torch import AWingLoss, WeightedAWingLoss


@pytest.fixture(scope='module')
def loss_fn():
    return AWingLoss()


@pytest.fixture(scope='module')
def weighted_loss_fn():
    return WeightedAWingLoss()


def test_AWingLoss():
    loss_fn = AWingLoss(alpha=2.1, omega=14, epsilon=1, theta=0.5)
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 2.0, 2.0])
    loss = loss_fn(preds, targets)
    assert loss.shape == preds.shape
    assert torch.allclose(loss, torch.tensor([9.9030, 0.0000, 9.9030]), atol=1e-4)


def test_WeightedAWingLoss():
    weighted_loss_fn = WeightedAWingLoss(w=10, alpha=2.1, omega=14, epsilon=1, theta=0.5)
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 2.0, 2.0])
    weight_map = torch.tensor([0, 1, 0], dtype=torch.bool)
    loss = weighted_loss_fn(preds, targets, weight_map=weight_map)
    assert torch.allclose(loss, torch.tensor(6.6020), atol=1e-4)

    # Test without weight_map
    loss = weighted_loss_fn(preds, targets)
    assert torch.allclose(loss, torch.tensor(72.6221), atol=1e-4)

    # Test with float weight_map
    weight_map = torch.tensor([0.0, 1.0, 0.0])
    loss = weighted_loss_fn(preds, targets, weight_map=weight_map)
    assert torch.allclose(loss, torch.tensor(6.6020), atol=1e-4)
