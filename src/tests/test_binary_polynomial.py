import pytest
import torch
from src.simulated_bifurcation.polynomial import BinaryPolynomial


matrix = [
    [0, 1, -1],
    [1, 0, 2],
    [-1, 2, 0]
]
vector = [1, 2, -3]
constant = 1

def test_init_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    ising = binary_polynomial.to_ising()
    ising.computed_spins = torch.Tensor([[1, -1], [-1, 1], [1, 1]])
    assert torch.all(ising.J == torch.Tensor([
        [0, -.5, .5],
        [-.5, 0, -1],
        [.5, -1, 0]
    ]))
    assert torch.all(ising.h == torch.Tensor([.5, 2.5, -1]))
    assert torch.all(binary_polynomial.convert_spins(ising) == torch.Tensor([1, 0, 1]))

def test_call_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    assert binary_polynomial([1, 0, 1]) == -3
    with pytest.raises(ValueError):
        binary_polynomial([1, 2, 3])

def test_optimize_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    assert torch.all(torch.Tensor([1, 0, 1]) == binary_polynomial.optimize(verbose=False))
