import pytest
import torch

from src.simulated_bifurcation.polynomial import BinaryPolynomial

matrix = torch.Tensor(
    [
        [0, 1, -1],
        [1, 0, 2],
        [-1, 2, 0],
    ]
)
vector = torch.Tensor([1, 2, -3])
constant = 1


def test_init_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    ising = binary_polynomial.to_ising()
    ising.computed_spins = torch.Tensor([[1, -1], [-1, 1], [1, 1]])
    assert torch.equal(
        ising.J,
        torch.Tensor(
            [
                [0, -0.5, 0.5],
                [-0.5, 0, -1],
                [0.5, -1, 0],
            ]
        ),
    )
    assert torch.equal(ising.h, torch.Tensor([0.5, 2.5, -1]))
    assert torch.equal(binary_polynomial.convert_spins(ising), torch.Tensor([1, 0, 1]))


def test_call_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    assert binary_polynomial(torch.Tensor([1, 0, 1])) == -3
    with pytest.raises(ValueError):
        binary_polynomial(torch.Tensor([1, 2, 3]))


def test_optimize_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    assert torch.equal(
        binary_polynomial.optimize(verbose=False), torch.Tensor([1, 0, 1])
    )
