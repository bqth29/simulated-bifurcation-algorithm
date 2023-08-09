import pytest
import torch

from src.simulated_bifurcation.polynomial import IntegerPolynomial

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


def test_init_integer_polynomial():
    with pytest.raises(ValueError):
        IntegerPolynomial(matrix, vector, constant, 0)
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        IntegerPolynomial(matrix, vector, constant, 2.5)
    integer_polynomial = IntegerPolynomial(matrix, vector, constant, 2)
    ising = integer_polynomial.to_ising()
    assert integer_polynomial.convert_spins(ising) is None
    ising.computed_spins = torch.tensor(
        [
            [1, -1],
            [-1, 1],
            [1, 1],
            [-1, -1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        ising.J,
        torch.tensor(
            [
                [0, 0, -0.5, -1, 0.5, 1],
                [0, 0, -1, -2, 1, 2],
                [-0.5, -1, 0, 0, -1, -2],
                [-1, -2, 0, 0, -2, -4],
                [0.5, 1, -1, -2, 0, 0],
                [1, 2, -2, -4, 0, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        ising.h, torch.tensor([0.5, 1, 5.5, 11, 0, 0], dtype=torch.float32)
    )
    print(integer_polynomial.convert_spins(ising))
    assert torch.equal(
        integer_polynomial.convert_spins(ising),
        torch.tensor(
            [
                [1, 2],
                [1, 1],
                [2, 0],
            ],
            dtype=torch.float32,
        ),
    )


def test_call_integer_polynomial():
    integer_polynomial = IntegerPolynomial(matrix, vector, constant, 2)
    assert integer_polynomial(torch.tensor([2, 3, 0], dtype=torch.float32)) == 21
    with pytest.raises(ValueError):
        integer_polynomial(torch.tensor([1, 2, 8], dtype=torch.float32))


def test_optimize_integer_polynomial():
    integer_polynomial = IntegerPolynomial(matrix, vector, constant, 2)
    int_vars, value = integer_polynomial.optimize(verbose=False)
    assert torch.equal(int_vars, torch.tensor([3, 0, 3], dtype=torch.float32))
    assert value == -23.0
