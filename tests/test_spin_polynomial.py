import pytest
import torch

from src.simulated_bifurcation.polynomial import SpinPolynomial, SpinQuadraticPolynomial

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
    spin_polynomial = SpinQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )
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
    spin_polynomial = SpinQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )
    assert spin_polynomial(torch.tensor([1, -1, 1], dtype=torch.float32)) == -11
    with pytest.raises(ValueError):
        spin_polynomial(torch.tensor([1, 2, 3], dtype=torch.float32))


def test_optimize_spin_polynomial():
    spin_polynomial = SpinQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )
    spin_vars, value = spin_polynomial.optimize(verbose=False)
    assert torch.equal(spin_vars, torch.tensor([1, -1, 1], dtype=torch.float32))
    assert value == -11.0


def test_to():
    spin_polynomial = SpinQuadraticPolynomial(
        matrix, vector, constant, silence_deprecation_warning=True
    )

    def check_device_and_dtype(dtype: torch.dtype):
        assert spin_polynomial.dtype == dtype
        assert spin_polynomial.device == torch.device("cpu")
        assert spin_polynomial.matrix.dtype == dtype
        assert spin_polynomial.vector.dtype == dtype
        assert spin_polynomial.constant.dtype == dtype
        assert spin_polynomial.matrix.device == torch.device("cpu")
        assert spin_polynomial.vector.device == torch.device("cpu")
        assert spin_polynomial.constant.device == torch.device("cpu")

    check_device_and_dtype(torch.float32)

    spin_polynomial.to(dtype=torch.float16)
    check_device_and_dtype(torch.float16)

    spin_polynomial.to(device="cpu")
    check_device_and_dtype(torch.float16)

    spin_polynomial.to(device="cpu", dtype=torch.float64)
    check_device_and_dtype(torch.float64)

    spin_polynomial.to()
    check_device_and_dtype(torch.float64)


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        SpinPolynomial(matrix, vector, constant)
