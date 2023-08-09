import pytest
import torch

from src.simulated_bifurcation.polynomial import SpinPolynomial

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


def test_init_spin_polynomial():
    spin_polynomial = SpinPolynomial(matrix, vector, constant)
    ising = spin_polynomial.to_ising()
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
                [0, -2, 2],
                [-2, 0, -4],
                [2, -4, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(ising.h, torch.tensor([1, 2, -3], dtype=torch.float32))
    assert torch.equal(
        spin_polynomial.convert_spins(ising),
        torch.tensor(
            [
                [1, -1],
                [-1, 1],
                [1, 1],
            ],
            dtype=torch.float32,
        ),
    )


def test_call_spin_polynomial():
    spin_polynomial = SpinPolynomial(matrix, vector, constant)
    assert spin_polynomial(torch.tensor([1, -1, 1], dtype=torch.float32)) == -11
    with pytest.raises(ValueError):
        spin_polynomial(torch.tensor([1, 2, 3], dtype=torch.float32))


def test_optimize_spin_polynomial():
    spin_polynomial = SpinPolynomial(matrix, vector, constant)
    spin_vars, value = spin_polynomial.optimize(verbose=False)
    assert torch.equal(spin_vars, torch.tensor([1, -1, 1], dtype=torch.float32))
    assert value == -11.0
