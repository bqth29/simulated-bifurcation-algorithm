import pytest
import torch

from src.simulated_bifurcation.polynomial import BinaryPolynomial

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


def test_init_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    ising = binary_polynomial.to_ising()
    assert binary_polynomial.convert_spins(ising) is None
    ising.computed_spins = torch.tensor(
        [
            [1, -1],
            [-1, 1],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        ising.J,
        torch.tensor(
            [
                [0, -0.5, 0.5],
                [-0.5, 0, -1],
                [0.5, -1, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(ising.h, torch.tensor([0.5, 2.5, -1], dtype=torch.float32))
    assert torch.equal(
        binary_polynomial.convert_spins(ising),
        torch.tensor(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=torch.float32,
        ),
    )


def test_call_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    assert binary_polynomial(torch.tensor([1, 0, 1], dtype=torch.float32)) == -3
    with pytest.raises(ValueError):
        binary_polynomial(torch.tensor([1, 2, 3], dtype=torch.float32))


def test_optimize_binary_polynomial():
    binary_polynomial = BinaryPolynomial(matrix, vector, constant)
    binary_vars, value = binary_polynomial.optimize(verbose=False)
    assert torch.equal(binary_vars, torch.tensor([1, 0, 1], dtype=torch.float32))
    assert value == -3.0
