import pytest
import torch

from src.simulated_bifurcation import maximize, minimize

matrix = torch.tensor(
    [
        [0, 1, -1],
        [1, 0, 2],
        [-1, 2, 0],
    ],
    dtype=torch.float32,
)
vector = torch.tensor([1, 2, -3], dtype=torch.float32)
constant = 1


def test_minimize_spin():
    best_combination, best_value = minimize(matrix, vector, constant, "spin")
    assert torch.equal(torch.tensor([1, -1, 1], dtype=torch.float32), best_combination)
    assert -11 == best_value


def test_minimize_binary():
    best_combination, best_value = minimize(matrix, vector, constant, "binary")
    assert torch.equal(torch.tensor([1, 0, 1], dtype=torch.float32), best_combination)
    assert -3 == best_value


def test_minimize_integer():
    best_combination, best_value = minimize(matrix, vector, constant, "int3")
    assert torch.equal(torch.tensor([7, 0, 7], dtype=torch.float32), best_combination)
    assert -111 == best_value


def test_maximize_spin():
    best_combination, best_value = maximize(matrix, vector, constant, "spin")
    assert torch.equal(
        best_combination, torch.tensor([1, -1, -1], dtype=torch.float32)
    ) or torch.equal(best_combination, torch.tensor([1, 1, -1], dtype=torch.float32))
    assert 7 == best_value


def test_maximize_binary():
    best_combination, best_value = maximize(matrix, vector, constant, "binary")
    assert torch.equal(torch.tensor([1, 1, 0], dtype=torch.float32), best_combination)
    assert 6 == best_value


def test_maximize_integer():
    best_combination, best_value = maximize(matrix, vector, constant, "int2")
    assert torch.equal(torch.tensor([3, 3, 3], dtype=torch.float32), best_combination)
    assert 37 == best_value


def test_wrong_input_value_type():
    with pytest.raises(ValueError):
        minimize(matrix, input_type="float")


def test_best_only():
    spins_best_only, energy_best_only = minimize(matrix, agents=42)
    assert spins_best_only.shape == (3,)
    assert isinstance(energy_best_only, float)
    spins_all, energies_all = minimize(matrix, agents=42, best_only=False)
    assert spins_all.shape == (42, 3)
    assert isinstance(energies_all, torch.Tensor)
    assert energies_all.shape == (42,)
