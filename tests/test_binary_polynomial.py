import pytest
import torch

from src.simulated_bifurcation.polynomial import (
    BinaryPolynomial,
    BinaryQuadraticPolynomial,
)

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
    binary_polynomial = BinaryQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )
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
    binary_polynomial = BinaryQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )
    assert binary_polynomial(torch.tensor([1, 0, 1], dtype=torch.float32)) == -3
    with pytest.raises(ValueError):
        binary_polynomial(torch.tensor([1, 2, 3], dtype=torch.float32))


def test_optimize_binary_polynomial():
    binary_polynomial = BinaryQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )
    binary_vars, value = binary_polynomial.optimize(verbose=False)
    assert torch.equal(binary_vars, torch.tensor([1, 0, 1], dtype=torch.float32))
    assert value == -3.0


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        BinaryQuadraticPolynomial(matrix, vector, constant)
    with pytest.warns(DeprecationWarning):
        BinaryPolynomial(matrix, vector, constant)
