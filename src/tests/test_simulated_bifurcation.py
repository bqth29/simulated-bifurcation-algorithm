import pytest
import torch

from src.simulated_bifurcation import maximize, minimize

matrix = torch.Tensor(
    [
        [0, 1, -1],
        [1, 0, 2],
        [-1, 2, 0],
    ]
)
vector = torch.Tensor([1, 2, -3])
constant = 1


def test_minimize_spin():
    best_combination, best_value = minimize(matrix, vector, constant, "spin")
    assert torch.equal(torch.Tensor([1, -1, 1]), best_combination)
    assert -11 == best_value


def test_minimize_binary():
    best_combination, best_value = minimize(matrix, vector, constant, "binary")
    assert torch.equal(torch.Tensor([1, 0, 1]), best_combination)
    assert -3 == best_value


def test_minimize_integer():
    best_combination, best_value = minimize(matrix, vector, constant, "int3")
    assert torch.equal(torch.Tensor([7, 0, 7]), best_combination)
    assert -111 == best_value


def test_maximize_spin():
    best_combination, best_value = maximize(matrix, vector, constant, "spin")
    assert torch.equal(torch.Tensor([1, 1, -1]), best_combination)
    assert 7 == best_value


def test_maximize_binary():
    best_combination, best_value = maximize(matrix, vector, constant, "binary")
    assert torch.equal(torch.Tensor([1, 1, 0]), best_combination)
    assert 6 == best_value


def test_maximize_integer():
    best_combination, best_value = maximize(matrix, vector, constant, "int2")
    assert torch.equal(torch.Tensor([3, 3, 3]), best_combination)
    assert 37 == best_value


def test_wrong_input_value_type():
    with pytest.raises(ValueError):
        minimize(matrix, input_type="float")
