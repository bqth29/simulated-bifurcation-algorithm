import pytest
import torch
from src.simulated_bifurcation.polynomial import SpinPolynomial


matrix = [
    [0, 1, -1],
    [1, 0, 2],
    [-1, 2, 0]
]
vector = [1, 2, -3]
constant = 1

def test_init_spin_polynomial():
    spin_polynomial = SpinPolynomial(matrix, vector, constant)
    ising = spin_polynomial.to_ising()
    ising.computed_spins = torch.Tensor([[1, -1], [-1, 1], [1, 1]])
    assert torch.all(ising.J == torch.Tensor([
        [0, -2, 2],
        [-2, 0, -4],
        [2, -4, 0]
    ]))
    assert torch.all(ising.h == torch.Tensor([1, 2, -3]))
    assert torch.all(spin_polynomial.convert_spins(ising) == torch.Tensor([1, -1, 1]))

def test_call_spin_polynomial():
    spin_polynomial = SpinPolynomial(matrix, vector, constant)
    assert spin_polynomial([1, -1, 1]) == -11
    with pytest.raises(ValueError):
        spin_polynomial([1, 2, 3])

def test_optimize_spin_polynomial():
    spin_polynomial = SpinPolynomial(matrix, vector, constant)
    assert torch.all(torch.Tensor([1, -1, 1]) == spin_polynomial.optimize(verbose=False))
