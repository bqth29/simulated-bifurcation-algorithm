from typing import Union

import numpy as np
import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.core import Ising, QuadraticPolynomial

DEVICES = (
    [torch.device("cpu"), torch.device("cuda")]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)
DTYPES = [torch.float32, torch.float64, torch.int32, torch.int64]

x, y, z = symbols("x y z")


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_from_poly(dtype: torch.dtype, device: torch.device):
    polynomial = poly(
        x**2 + 3 * y**2 - 5 * z**2 + 4 * x * y - 2 * y * z + 8 * x - 7 * y + z + 6
    )
    quadratic_polynomial = QuadraticPolynomial(polynomial, dtype=dtype, device=device)

    assert quadratic_polynomial._dtype == dtype
    assert quadratic_polynomial._device == device
    assert quadratic_polynomial._n_gens == 3
    assert quadratic_polynomial._poly == polynomial
    assert torch.equal(
        torch.tensor(
            [[1, 4, 0], [0, 3, -2], [0, 0, -5]],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[2],
    )
    assert torch.equal(
        torch.tensor(
            [8, -7, 1],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[1],
    )
    assert torch.equal(
        torch.tensor(6, dtype=dtype, device=device),
        quadratic_polynomial[0],
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_from_poly_no_bias(dtype: torch.dtype, device: torch.device):
    polynomial = poly(
        x**2 + 3 * y**2 - 5 * z**2 + 4 * x * y - 2 * y * z + 8 * x - 7 * y + z
    )
    quadratic_polynomial = QuadraticPolynomial(polynomial, dtype=dtype, device=device)

    assert quadratic_polynomial._dtype == dtype
    assert quadratic_polynomial._device == device
    assert quadratic_polynomial._n_gens == 3
    assert quadratic_polynomial._poly == polynomial
    assert torch.equal(
        torch.tensor(
            [[1, 4, 0], [0, 3, -2], [0, 0, -5]],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[2],
    )
    assert torch.equal(
        torch.tensor(
            [8, -7, 1],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[1],
    )
    assert torch.equal(
        torch.tensor(0, dtype=dtype, device=device),
        quadratic_polynomial[0],
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_from_poly_no_degree_1_monoms(dtype: torch.dtype, device: torch.device):
    polynomial = poly(x**2 + 3 * y**2 - 5 * z**2 + 4 * x * y - 2 * y * z + 6)
    quadratic_polynomial = QuadraticPolynomial(polynomial, dtype=dtype, device=device)

    assert quadratic_polynomial._dtype == dtype
    assert quadratic_polynomial._device == device
    assert quadratic_polynomial._n_gens == 3
    assert quadratic_polynomial._poly == polynomial
    assert torch.equal(
        torch.tensor(
            [[1, 4, 0], [0, 3, -2], [0, 0, -5]],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[2],
    )
    assert torch.equal(
        torch.tensor(
            [0, 0, 0],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[1],
    )
    assert torch.equal(
        torch.tensor(6, dtype=dtype, device=device),
        quadratic_polynomial[0],
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_from_poly_no_degree_2_monoms(dtype: torch.dtype, device: torch.device):
    polynomial = poly(8 * x - 7 * y + z + 6)
    quadratic_polynomial = QuadraticPolynomial(polynomial, dtype=dtype, device=device)

    assert quadratic_polynomial._dtype == dtype
    assert quadratic_polynomial._device == device
    assert quadratic_polynomial._n_gens == 3
    assert quadratic_polynomial._poly == polynomial
    assert torch.equal(
        torch.tensor(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[2],
    )
    assert torch.equal(
        torch.tensor(
            [8, -7, 1],
            dtype=dtype,
            device=device,
        ),
        quadratic_polynomial[1],
    )
    assert torch.equal(
        torch.tensor(6, dtype=dtype, device=device),
        quadratic_polynomial[0],
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_init_from_poly_with_degree_greater_than_2(
    dtype: torch.dtype, device: torch.device
):
    with pytest.raises(
        ValueError,
        match="Expected a quadratic polynomial but got a total degree of 3.",
    ):
        QuadraticPolynomial(poly(x**3), dtype=dtype, device=device)


## From tensors


def make_quadratic_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=dtype, device=device
    )


def make_linear_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor([-1, -2, 1], dtype=dtype, device=device)


def make_bias(
    as_tensor: bool, dtype: torch.dtype, device: torch.device
) -> Union[float, torch.Tensor]:
    return torch.tensor(2, dtype=dtype, device=device) if as_tensor else 2.0


@pytest.mark.parametrize(
    "bias_as_tensor, dtype, device",
    [
        (bias_as_tensor, dtype, device)
        for bias_as_tensor in [True, False]
        for dtype in DTYPES
        for device in DEVICES
    ],
)
def test_build_polynomial_from_tensor(
    bias_as_tensor: bool, dtype: torch.dtype, device: torch.device
):
    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device), dtype=dtype, device=device
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(torch.zeros(3, dtype=dtype, device=device), polynomial[1])
    assert torch.equal(torch.tensor(0, dtype=dtype, device=device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device),
        make_linear_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(torch.tensor(0, dtype=dtype, device=device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_linear_tensor(dtype, device),
        make_quadratic_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(torch.tensor(0, dtype=dtype, device=device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(torch.zeros(3, dtype=dtype, device=device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_bias(bias_as_tensor, dtype, device),
        make_quadratic_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(torch.zeros(3, dtype=dtype, device=device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_linear_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(torch.zeros(3, 3, dtype=dtype, device=device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_bias(bias_as_tensor, dtype, device),
        make_linear_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(torch.zeros(3, 3, dtype=dtype, device=device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device),
        make_linear_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device),
        make_linear_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        make_linear_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_quadratic_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        make_linear_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_linear_tensor(dtype, device),
        make_quadratic_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_linear_tensor(dtype, device),
        make_quadratic_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_linear_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        make_quadratic_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_linear_tensor(dtype, device),
        make_bias(bias_as_tensor, dtype, device),
        make_quadratic_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_bias(bias_as_tensor, dtype, device),
        make_linear_tensor(dtype, device),
        make_quadratic_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_bias(bias_as_tensor, dtype, device),
        make_linear_tensor(dtype, device),
        make_quadratic_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_bias(bias_as_tensor, dtype, device),
        make_quadratic_tensor(dtype, device),
        make_linear_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    polynomial = QuadraticPolynomial(
        make_bias(bias_as_tensor, dtype, device),
        make_quadratic_tensor(dtype, device),
        make_linear_tensor(dtype, device),
        dtype=dtype,
        device=device,
    )
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(make_quadratic_tensor(dtype, device), polynomial[2])
    assert torch.equal(make_linear_tensor(dtype, device), polynomial[1])
    assert torch.equal(make_bias(True, dtype, device), polynomial[0])

    with pytest.raises(
        ValueError,
        match="Cannot provide two quadratic coefficients tensors.",
    ):
        QuadraticPolynomial(
            make_quadratic_tensor(dtype, device), make_quadratic_tensor(dtype, device)
        )

    with pytest.raises(
        ValueError,
        match="Cannot provide two linear coefficients tensors.",
    ):
        QuadraticPolynomial(
            make_linear_tensor(dtype, device), make_linear_tensor(dtype, device)
        )

    with pytest.raises(
        ValueError,
        match="Cannot provide two biases.",
    ):
        QuadraticPolynomial(
            make_bias(bias_as_tensor, dtype, device),
            make_bias(bias_as_tensor, dtype, device),
        )

    with pytest.raises(
        ValueError, match="Expected a tensor with at most 2 dimension, got 3."
    ):
        QuadraticPolynomial(
            torch.zeros((3, 3, 3), dtype=dtype, device=device),
            make_quadratic_tensor(dtype, device),
            make_linear_tensor(dtype, device),
            2,
            dtype=torch.float32,
        )

    with pytest.raises(
        ValueError,
        match="Quadratic coefficients tensor is not symmetrical.",
    ):
        QuadraticPolynomial(torch.zeros((3, 2), dtype=dtype, device=device))

    with pytest.raises(
        ValueError,
        match="Unconsistant shape among provided tensors. Expected 3 but got 2.",
    ):
        QuadraticPolynomial(
            torch.zeros(3, 3, dtype=dtype, device=device),
            torch.zeros(2, dtype=dtype, device=device),
        )

    with pytest.raises(
        ValueError,
        match="Unconsistant shape among provided tensors. Expected 2 but got 3.",
    ):
        QuadraticPolynomial(
            torch.zeros(2, dtype=dtype, device=device),
            torch.zeros(3, 3, dtype=dtype, device=device),
        )


def test_build_polynomial_with_wrong_domain():
    with pytest.raises(
        ValueError,
        match="Unsupported coefficient tensor type: <class 'str'>. Expected a torch.Tensor or a numpy.ndarray.",
    ):
        QuadraticPolynomial("Hello world!")


matrix = torch.tensor(
    [
        [0, 1, -1],
        [1, 0, 2],
        [-1, 2, 0],
    ],
    dtype=torch.float32,
)
vector = torch.tensor([1, 2, -3], dtype=torch.float32)
constant_int = 1


def test_init_spin_polynomial():
    polynomial = QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32)
    ising = polynomial.to_ising(domain="spin")
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
        polynomial.convert_spins(ising, domain="spin"),
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
    polynomial = QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32)
    assert polynomial(torch.tensor([1, -1, 1], dtype=torch.float32)) == -11


def test_optimize_spin_polynomial():
    polynomial = QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32)
    spin_vars, value = polynomial.optimize(domain="spin", verbose=False)
    assert torch.equal(spin_vars, torch.tensor([1, -1, 1], dtype=torch.float32))
    assert value == -11.0


def test_minimize_spin_polynomial():
    torch.manual_seed(42)
    polynomial = QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32)
    spin_vars, value = polynomial.minimize(domain="spin", verbose=False)
    assert torch.equal(spin_vars, torch.tensor([1, -1, 1], dtype=torch.float32))
    assert value == -11.0


def test_maximize_spin_polynomial():
    torch.manual_seed(42)
    polynomial = QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32)
    spin_vars, value = polynomial.maximize(domain="spin", verbose=False)
    assert torch.equal(spin_vars, torch.tensor([1, 1, -1], dtype=torch.float32))
    assert value == 7.0


def test_init_binary_polynomial():
    binary_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    ising = binary_polynomial.to_ising(domain="binary")
    assert binary_polynomial.convert_spins(ising, domain="binary") is None
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
        binary_polynomial.convert_spins(ising, domain="binary"),
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
    binary_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    assert binary_polynomial(torch.tensor([1, 0, 1], dtype=torch.float32)) == -3


def test_optimize_binary_polynomial():
    binary_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    binary_vars, value = binary_polynomial.optimize(domain="binary", verbose=False)
    assert torch.equal(binary_vars, torch.tensor([1, 0, 1], dtype=torch.float32))
    assert value == -3.0


def test_init_integer_polynomial():
    integer_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    ising = integer_polynomial.to_ising(domain="int2")
    assert integer_polynomial.convert_spins(ising, domain="int2") is None
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
    assert torch.equal(
        integer_polynomial.convert_spins(ising, domain="int2"),
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
    integer_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    assert integer_polynomial(torch.tensor([2, 3, 0], dtype=torch.float32)) == 21


def test_optimize_integer_polynomial():
    integer_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    int_vars, value = integer_polynomial.optimize(domain="int2", verbose=False)
    assert torch.equal(int_vars, torch.tensor([3, 0, 3], dtype=torch.float32))
    assert value == -23.0


# def test_wrong_domain_to_ising():
#     with pytest.raises(ValueError):
#         QuadraticPolynomial(quadratic).to_ising(domain="int2.5")


# def test_wrong_domain_convert_spin():
#     ising = Ising(quadratic)
#     ising.computed_spins = torch.ones(3, 3)
#     with pytest.raises(ValueError):
#         QuadraticPolynomial(quadratic).convert_spins(ising, domain="Hello world!")


# def test_evaluate_polynomial():
#     polynomial = QuadraticPolynomial(
#         make_quadratic_tensor(dtype, device), dtype=torch.float32
#     )
#     data = [[0, 1, 0], [1, 0, 1]]
#     assert torch.equal(
#         torch.tensor([2, 3], dtype=torch.float32),
#         polynomial(torch.tensor(data, dtype=torch.float32)),
#     )
#     assert torch.equal(
#         torch.tensor([2, 3], dtype=torch.float32),
#         polynomial(np.array(data, dtype=np.float32)),
#     )
#     with pytest.raises(TypeError, match="Input value cannot be cast to Tensor."):
#         polynomial("Hello world!")
#     with pytest.raises(
#         ValueError, match="Size of the input along the last axis should be 3, it is 5."
#     ):
#         polynomial(torch.zeros(3, 3, 5))
