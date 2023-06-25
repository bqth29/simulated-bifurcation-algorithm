import torch
import numpy as np
import pytest
from src.simulated_bifurcation import Ising


J = [[1, 2, 3], [2, 1, 4], [3, 4, 1]]
h = [1, 0, -1]

def test_init_ising_model_from_tensors():
    ising = Ising(torch.Tensor(J), torch.Tensor(h))
    assert torch.all(ising.J == torch.Tensor(J))
    assert torch.all(ising.h == torch.Tensor(h))
    assert ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3

def test_init_ising_model_from_arrays():
    ising = Ising(np.array(J), np.array(h))
    assert torch.all(ising.J == torch.Tensor(J))
    assert torch.all(ising.h == torch.Tensor(h))
    assert ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3

def test_ising_without_linear_term():
    ising = Ising(torch.Tensor(J))
    assert torch.all(ising.J == torch.Tensor(J))
    assert torch.all(ising.h == torch.zeros(3))
    assert not ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3

def test_init_ising_with_null_h_vector():
    ising = Ising(torch.Tensor(J), torch.zeros(3))
    assert torch.all(ising.J == torch.Tensor(J))
    assert torch.all(ising.h == torch.zeros(3))
    assert not ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3

def test_attach():
    attached = Ising.attach(torch.Tensor(J), torch.Tensor(h))
    assert torch.all(attached == torch.Tensor([[1, 2, 3, -1], [2, 1, 4, 0], [3, 4, 1, 1], [-1, 0, 1, 0]]))

def test_call():
    ising = Ising(np.array(J), np.array(h))
    assert ising(None) is None
    assert ising(torch.Tensor([1, 1, -1])) == 5.5
    assert ising(torch.Tensor([[1], [1], [-1]])) == 5.5
    assert ising(torch.Tensor([[1, 1, -1]])) == 5.5
    assert ising(torch.Tensor([[1, -1], [1, 1], [-1, -1]])) == [5.5, 1.5]
    assert ising(np.array([1, 1, -1])) == 5.5
    assert ising(np.array([[1], [1], [-1]])) == 5.5
    assert ising(np.array([[1, 1, -1]])) == 5.5
    assert ising(np.array([[1, -1], [1, 1], [-1, -1]])) == [5.5, 1.5]
    with pytest.raises(TypeError):
        ising([1, 1, -1])
    with pytest.raises(ValueError):
        ising(torch.Tensor([1, 2, -1]))
    with pytest.raises(ValueError):
        ising(torch.Tensor([1, 1, -1, 1]))

def test_matrix_format():
    original = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_result = torch.Tensor([[0, 3, 5], [3, 0, 7], [5, 7, 0]])
    assert torch.all(expected_result == Ising.format_matrix(original))

def test_min():
    ising = Ising(np.array(J), np.array(h))
    spins = torch.Tensor([[1, -1], [1, 1], [-1, -1]])
    assert torch.all(ising.min(spins) == torch.Tensor([-1, 1, -1]))
