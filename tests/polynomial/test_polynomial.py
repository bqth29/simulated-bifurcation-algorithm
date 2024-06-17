import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.polynomial.polynomial import Polynomial

from ..utils import DEVICES, DTYPES


def assert_is_expected_polynomial(
    polynomial: Polynomial, dtype: torch.dtype, device: str
):
    """
    Assert that the polynomial corresponds to
    x**2 - 8 * x * y + 4 * y**2 + 6 * x - 9 * y - 3
    """
    assert polynomial.degree == 2
    assert polynomial.n_variables == 2
    assert polynomial.dtype == dtype
    assert polynomial.device == torch.device(device)
    assert torch.equal(
        torch.tensor([[1, -8], [0, 4]], dtype=dtype, device=device), polynomial[2]
    )
    assert torch.equal(torch.tensor([6, -9], dtype=dtype, device=device), polynomial[1])
    assert torch.equal(torch.tensor(-3, dtype=dtype, device=device), polynomial[0])


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_polynomial_from_tensors_sequence(dtype: torch.dtype, device: str):
    polynomial = Polynomial(
        torch.tensor([[1, -8], [0, 4]]),
        torch.tensor([6, -9]),
        -3,
        dtype=dtype,
        device=device,
    )
    assert_is_expected_polynomial(polynomial, dtype, device)


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_polynomial_from_expression(dtype: torch.dtype, device: str):
    x, y = symbols("x y")
    polynomial = Polynomial(
        poly(x**2 - 8 * x * y + 4 * y**2 + 6 * x - 9 * y - 3),
        dtype=dtype,
        device=device,
    )
    assert_is_expected_polynomial(polynomial, dtype, device)


@pytest.mark.parametrize(
    "original_dtype, target_dtype, original_device, target_device",
    [
        (original_dtype, target_dtype, original_device, target_device)
        for original_dtype in DTYPES
        for target_dtype in DTYPES
        for original_device in DEVICES
        for target_device in DEVICES
    ],
)
def test_migrate_polynomial(
    original_dtype: torch.dtype,
    target_dtype: torch.dtype,
    original_device: str,
    target_device: str,
):
    polynomial = Polynomial(
        torch.tensor([[1, -8], [0, 4]]),
        torch.tensor([6, -9]),
        -3,
        dtype=original_dtype,
        device=original_device,
    )
    polynomial.to(dtype=target_dtype, device=target_device)
    assert_is_expected_polynomial(polynomial, target_dtype, target_device)
