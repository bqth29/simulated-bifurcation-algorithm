import pytest
import torch

from src.simulated_bifurcation.polynomial import IntegerPolynomial

matrix = [
    [0, 1, -1],
    [1, 0, 2],
    [-1, 2, 0],
]
vector = [1, 2, -3]
constant = 1


def test_init_integer_polynomial():
    with pytest.raises(ValueError):
        IntegerPolynomial(matrix, vector, constant, 0)
    with pytest.raises(ValueError):
        IntegerPolynomial(matrix, vector, constant, 2.5)
    integer_polynomial = IntegerPolynomial(matrix, vector, constant, 2)
    ising = integer_polynomial.to_ising()
    ising.computed_spins = torch.Tensor(
        [[1, -1], [-1, 1], [1, 1], [-1, -1], [-1, -1], [1, -1]]
    )
    assert torch.equal(
        ising.J,
        torch.Tensor(
            [
                [0, 0, -0.5, -1, 0.5, 1],
                [0, 0, -1, -2, 1, 2],
                [-0.5, -1, 0, 0, -1, -2],
                [-1, -2, 0, 0, -2, -4],
                [0.5, 1, -1, -2, 0, 0],
                [1, 2, -2, -4, 0, 0],
            ]
        ),
    )
    assert torch.equal(ising.h, torch.Tensor([0.5, 1, 5.5, 11, 0, 0]))
    assert torch.equal(integer_polynomial.convert_spins(ising), torch.Tensor([1, 1, 2]))


def test_call_integer_polynomial():
    integer_polynomial = IntegerPolynomial(matrix, vector, constant, 2)
    assert integer_polynomial([2, 3, 0]) == 21
    with pytest.raises(ValueError):
        integer_polynomial([1, 2, 8])


def test_optimize_integer_polynomial():
    integer_polynomial = IntegerPolynomial(matrix, vector, constant, 2)
    assert torch.equal(
        integer_polynomial.optimize(verbose=False), torch.Tensor([3, 0, 3])
    )
