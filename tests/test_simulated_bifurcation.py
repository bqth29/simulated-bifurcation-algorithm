import pytest
import torch

import src.simulated_bifurcation
from src.simulated_bifurcation import build_model, maximize, minimize, optimize

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


def test_valid_domain():
    build_model(matrix, domain="spin")
    build_model(matrix, domain="binary")
    build_model(matrix, domain="int1")
    build_model(matrix, domain="int3")
    build_model(matrix, domain="int10")
    build_model(matrix, domain="int22")


def test_invalid_domain():
    with pytest.raises(ValueError):
        build_model(matrix, domain="float")
    with pytest.raises(ValueError):
        build_model(matrix, domain="")
    with pytest.raises(ValueError):
        build_model(matrix, domain="int")
    with pytest.raises(ValueError):
        build_model(matrix, domain=" int3")
    with pytest.raises(ValueError):
        build_model(matrix, domain="int0")
    with pytest.raises(ValueError):
        build_model(matrix, domain="int07")
    with pytest.raises(ValueError):
        build_model(matrix, domain="int5.")


def test_best_only():
    spins_best_only, energy_best_only = minimize(matrix, agents=42)
    assert spins_best_only.shape == (3,)
    assert isinstance(energy_best_only, torch.Tensor)
    assert energy_best_only.shape == ()
    spins_all, energies_all = minimize(matrix, agents=42, best_only=False)
    assert spins_all.shape == (42, 3)
    assert isinstance(energies_all, torch.Tensor)
    assert energies_all.shape == (42,)


def test_input_type_deprecation():
    with pytest.warns(DeprecationWarning):
        optimize(matrix, vector, constant, input_type="int7")
    with pytest.warns(DeprecationWarning):
        minimize(matrix, vector, constant, input_type="spin")
    with pytest.warns(DeprecationWarning):
        maximize(matrix, vector, constant, input_type="binary")
    with pytest.warns(DeprecationWarning):
        model = build_model(matrix, vector, constant, input_type="int3")
    assert isinstance(model, src.simulated_bifurcation.IntegerQuadraticPolynomial)
