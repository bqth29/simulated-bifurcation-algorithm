import torch
from sympy import poly, symbols

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

x, y, z = symbols("x y z")
expression = poly(
    x * y - x * z + y * x + 2 * y * z - z * x + 2 * z * y + x + 2 * y - 3 * z + 1
)


def test_minimize_spin():
    best_combination, best_value = minimize(
        matrix, vector, constant, domain="spin", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([1, -1, 1], dtype=torch.float32), best_combination)
    assert -11 == best_value
    best_combination, best_value = minimize(
        expression, domain="spin", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([1, -1, 1], dtype=torch.float32), best_combination)
    assert -11 == best_value


def test_minimize_binary():
    best_combination, best_value = minimize(
        matrix, vector, constant, domain="binary", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([1, 0, 1], dtype=torch.float32), best_combination)
    assert -3 == best_value
    best_combination, best_value = minimize(
        expression, domain="binary", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([1, 0, 1], dtype=torch.float32), best_combination)
    assert -3 == best_value


def test_minimize_integer():
    best_combination, best_value = minimize(
        matrix, vector, constant, domain="int3", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([7, 0, 7], dtype=torch.float32), best_combination)
    assert -111 == best_value
    best_combination, best_value = minimize(
        expression, domain="int3", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([7, 0, 7], dtype=torch.float32), best_combination)
    assert -111 == best_value


def test_maximize_spin():
    best_combination, best_value = maximize(
        matrix, vector, constant, domain="spin", dtype=torch.float32
    )
    assert torch.equal(
        best_combination, torch.tensor([1, -1, -1], dtype=torch.float32)
    ) or torch.equal(best_combination, torch.tensor([1, 1, -1], dtype=torch.float32))
    assert 7 == best_value
    best_combination, best_value = maximize(
        expression, domain="spin", dtype=torch.float32
    )
    assert torch.equal(
        best_combination, torch.tensor([1, -1, -1], dtype=torch.float32)
    ) or torch.equal(best_combination, torch.tensor([1, 1, -1], dtype=torch.float32))
    assert 7 == best_value


def test_maximize_binary():
    best_combination, best_value = maximize(
        matrix, vector, constant, domain="binary", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([1, 1, 0], dtype=torch.float32), best_combination)
    assert 6 == best_value
    best_combination, best_value = maximize(
        expression, domain="binary", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([1, 1, 0], dtype=torch.float32), best_combination)
    assert 6 == best_value


def test_maximize_integer():
    best_combination, best_value = maximize(
        matrix, vector, constant, domain="int2", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([3, 3, 3], dtype=torch.float32), best_combination)
    assert 37 == best_value
    best_combination, best_value = maximize(
        expression, domain="int2", dtype=torch.float32
    )
    assert torch.equal(torch.tensor([3, 3, 3], dtype=torch.float32), best_combination)
    assert 37 == best_value


def test_best_only():
    spins_best_only, energy_best_only = minimize(
        matrix, domain="spin", agents=42, dtype=torch.float32
    )
    assert spins_best_only.shape == (3,)
    assert isinstance(energy_best_only, torch.Tensor)
    assert energy_best_only.shape == ()
    spins_all, energies_all = minimize(
        matrix, domain="spin", agents=42, best_only=False, dtype=torch.float32
    )
    assert spins_all.shape == (42, 3)
    assert isinstance(energies_all, torch.Tensor)
    assert energies_all.shape == (42,)
