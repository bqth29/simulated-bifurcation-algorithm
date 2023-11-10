import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.core import (
    Ising,
    QuadraticPolynomial,
    QuadraticPolynomialError,
)
from src.simulated_bifurcation.polynomial.polynomial_map import (
    PolynomialMapTensorDimensionError,
)

quadratic = torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]])
linear = torch.tensor([-1, -2, 1])
constant = torch.tensor(2)


def test_build_polynomial_from_expression():
    x, y, z = symbols("x y z")
    expression = poly(
        x**2
        + 2 * y**2
        + 3 * z**2
        - 2 * x * y
        - x * z
        - 3 * y * z
        - x
        - 2 * y
        + z
        + 2
    )

    # Valid definitions

    polynomial = QuadraticPolynomial(expression)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    # Not quadratic polynomials

    with pytest.raises(
        QuadraticPolynomialError, match="Expected a degree 2 polynomial, got 1."
    ):
        QuadraticPolynomial(poly(x + y + z))

    with pytest.raises(
        QuadraticPolynomialError, match="Expected a degree 2 polynomial, got 3."
    ):
        QuadraticPolynomial(poly(x**3))


def test_build_polynomial_from_tensor():
    polynomial = QuadraticPolynomial(quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(torch.tensor(0), polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, linear)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(torch.tensor(0), polynomial[0])

    polynomial = QuadraticPolynomial(linear, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(torch.tensor(0), polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, 2)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(2, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, constant)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(constant, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, linear, constant)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, linear, 2)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, constant, linear)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, 2, linear)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, quadratic, constant)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, quadratic, 2)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, constant, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, 2, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(constant, linear, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(2, linear, quadratic)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(constant, quadratic, linear)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(2, quadratic, linear)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    with pytest.raises(
        ValueError,
        match="Declaring twice the same degree is ambiguous. Got two tensors for degree 2.",
    ):
        QuadraticPolynomial(quadratic, quadratic)

    with pytest.raises(
        QuadraticPolynomialError, match="Expected a degree 2 polynomial, got 1."
    ):
        QuadraticPolynomial(linear)

    with pytest.raises(
        QuadraticPolynomialError, match="Expected a degree 2 polynomial, got 3."
    ):
        QuadraticPolynomial(torch.zeros((3, 3, 3)), quadratic, linear, 2)

    with pytest.raises(
        PolynomialMapTensorDimensionError,
        match="All dimensions of the tensor must be equal.",
    ):
        QuadraticPolynomial(torch.zeros((3, 2)))


def test_build_polynomial_with_wrong_input_type():
    with pytest.raises(
        TypeError,
        match="Expected tensors or arrays, got <class 'str'>.",
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
    polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    ising = polynomial.to_ising(input_type="spin")
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
        polynomial.convert_spins(ising, input_type="spin"),
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
    polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    assert polynomial(torch.tensor([1, -1, 1], dtype=torch.float32)) == -11


def test_optimize_spin_polynomial():
    polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    spin_vars, value = polynomial.optimize(input_type="spin", verbose=False)
    assert torch.equal(spin_vars, torch.tensor([1, -1, 1], dtype=torch.float32))
    assert value == -11.0


def test_init_binary_polynomial():
    binary_polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    ising = binary_polynomial.to_ising(input_type="binary")
    assert binary_polynomial.convert_spins(ising, input_type="binary") is None
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
        binary_polynomial.convert_spins(ising, input_type="binary"),
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
    binary_polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    assert binary_polynomial(torch.tensor([1, 0, 1], dtype=torch.float32)) == -3


def test_optimize_binary_polynomial():
    binary_polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    binary_vars, value = binary_polynomial.optimize(input_type="binary", verbose=False)
    assert torch.equal(binary_vars, torch.tensor([1, 0, 1], dtype=torch.float32))
    assert value == -3.0


def test_init_integer_polynomial():
    integer_polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    ising = integer_polynomial.to_ising(input_type="int2")
    assert integer_polynomial.convert_spins(ising, input_type="int2") is None
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
        integer_polynomial.convert_spins(ising, input_type="int2"),
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
    integer_polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    assert integer_polynomial(torch.tensor([2, 3, 0], dtype=torch.float32)) == 21


def test_optimize_integer_polynomial():
    integer_polynomial = QuadraticPolynomial(matrix, vector, constant_int)
    int_vars, value = integer_polynomial.optimize(input_type="int2", verbose=False)
    assert torch.equal(int_vars, torch.tensor([3, 0, 3], dtype=torch.float32))
    assert value == -23.0


def test_to():
    polynomial = QuadraticPolynomial(matrix, vector, constant_int)

    def check_device_and_dtype(dtype: torch.dtype):
        assert polynomial.dtype == dtype
        assert polynomial.device == torch.device("cpu")
        assert polynomial[2].dtype == dtype
        assert polynomial[1].dtype == dtype
        assert polynomial[0].dtype == dtype
        assert polynomial[2].device == torch.device("cpu")
        assert polynomial[1].device == torch.device("cpu")
        assert polynomial[0].device == torch.device("cpu")

    check_device_and_dtype(torch.float32)

    polynomial.to(dtype=torch.float16)
    check_device_and_dtype(torch.float16)

    polynomial.to(device="cpu")
    check_device_and_dtype(torch.float16)

    polynomial.to(device="cpu", dtype=torch.float64)
    check_device_and_dtype(torch.float64)

    polynomial.to()
    check_device_and_dtype(torch.float64)


def test_wrong_input_type_to_ising():
    with pytest.raises(ValueError):
        QuadraticPolynomial(quadratic).to_ising(input_type="int2.5")


def test_wrong_input_type_cnvert_spin():
    ising = Ising(quadratic)
    ising.computed_spins = torch.ones(3, 3)
    with pytest.raises(ValueError):
        QuadraticPolynomial(quadratic).convert_spins(ising, input_type="Hello world!")
