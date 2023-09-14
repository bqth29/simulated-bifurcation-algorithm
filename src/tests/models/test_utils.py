import pytest
import torch
from numpy import array

from src.simulated_bifurcation.models.utils import cast_to_tensor


def test_cast_tensor_to_tensor():
    assert torch.equal(torch.tensor([1, 2, 3]), cast_to_tensor(torch.tensor([1, 2, 3])))


def test_cast_array_to_tensor():
    assert torch.equal(torch.tensor([1, 2, 3]), cast_to_tensor(array([1, 2, 3])))


def test_attempt_to_cast_wrong_type():
    with pytest.raises(TypeError, match="Expected tensor or array, got int."):
        cast_to_tensor(1)
