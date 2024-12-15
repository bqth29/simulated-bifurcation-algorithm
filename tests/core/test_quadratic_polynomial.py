import numpy as np
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

quadratic = torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=torch.float32)
linear = torch.tensor([-1, -2, 1], dtype=torch.float32)
constant = torch.tensor(2, dtype=torch.float32)


def test_build_polynomial_from_expression():
    x, y, z = symbols("x y z")
    expression = poly(
        x**2 + 2 * y**2 + 3 * z**2 - 2 * x * y - x * z - 3 * y * z - x - 2 * y + z + 2
    )

    # Valid definitions

    polynomial = QuadraticPolynomial(expression, dtype=torch.float32)
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
    polynomial = QuadraticPolynomial(quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(torch.tensor(0), polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, linear, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(torch.tensor(0), polynomial[0])

    polynomial = QuadraticPolynomial(linear, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(torch.tensor(0), polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, 2, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(2, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, constant, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(constant, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(torch.zeros(3), polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, linear, constant, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, linear, 2, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, constant, linear, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(quadratic, 2, linear, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, quadratic, constant, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, quadratic, 2, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, constant, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(linear, 2, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(constant, linear, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(2, linear, quadratic, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(constant, quadratic, linear, dtype=torch.float32)
    assert isinstance(polynomial, QuadraticPolynomial)
    assert torch.equal(quadratic, polynomial[2])
    assert torch.equal(linear, polynomial[1])
    assert torch.equal(constant, polynomial[0])

    polynomial = QuadraticPolynomial(2, quadratic, linear, dtype=torch.float32)
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
        QuadraticPolynomial(
            torch.zeros((3, 3, 3)), quadratic, linear, 2, dtype=torch.float32
        )

    with pytest.raises(
        PolynomialMapTensorDimensionError,
        match="All dimensions of the tensor must be equal.",
    ):
        QuadraticPolynomial(torch.zeros((3, 2)))


def test_build_polynomial_with_wrong_domain():
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


def test_init_mixed_types_polynomial():
    integer_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    ising = integer_polynomial.to_ising(domain=["spin", "binary", "int2"])
    assert integer_polynomial.convert_spins(ising, domain="int2") is None
    ising.computed_spins = torch.tensor(
        [
            [1, -1],
            [-1, 1],
            [1, 1],
            [-1, -1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        ising.J,
        torch.tensor(
            [
                [0, -1, 1, 2],
                [-1, 0, -1, -2],
                [1, -1, 0, 0],
                [2, -2, 0, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(ising.h, torch.tensor([-1, 4, -0.5, -1], dtype=torch.float32))
    assert torch.equal(
        integer_polynomial.convert_spins(ising, domain=["spin", "binary", "int2"]),
        torch.tensor(
            [
                [1, -1],
                [0, 1],
                [1, 1],
            ],
            dtype=torch.float32,
        ),
    )


def test_call_mixed_types_polynomial():
    integer_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    assert integer_polynomial(torch.tensor([-1, 1, 3], dtype=torch.float32)) == 9.0


def test_optimize_mixed_types_polynomial():
    integer_polynomial = QuadraticPolynomial(
        matrix, vector, constant_int, dtype=torch.float32
    )
    int_vars, value = integer_polynomial.optimize(
        domain=["spin", "binary", "int2"], verbose=False
    )
    assert torch.equal(int_vars, torch.tensor([1, 0, 3], dtype=torch.float32))
    assert value == -13.0


def test_init_mixed_types_polynomial_wrong_number_of_domains():
    with pytest.raises(ValueError, match="Expected 3 domains to be provided, got 2."):
        QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32).to_ising(
            domain=["spin", "binary"]
        )


def test_to():
    polynomial = QuadraticPolynomial(matrix, vector, constant_int, dtype=torch.float32)

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


def test_wrong_domain_to_ising():
    with pytest.raises(ValueError):
        QuadraticPolynomial(quadratic).to_ising(domain="int2.5")


def test_wrong_domain_convert_spin():
    ising = Ising(quadratic)
    ising.computed_spins = torch.ones(3, 3)
    with pytest.raises(ValueError):
        QuadraticPolynomial(quadratic).convert_spins(ising, domain="Hello world!")


def test_evaluate_polynomial():
    polynomial = QuadraticPolynomial(quadratic, dtype=torch.float32)
    data = [[0, 1, 0], [1, 0, 1]]
    assert torch.equal(
        torch.tensor([2, 3], dtype=torch.float32),
        polynomial(torch.tensor(data, dtype=torch.float32)),
    )
    assert torch.equal(
        torch.tensor([2, 3], dtype=torch.float32),
        polynomial(np.array(data, dtype=np.float32)),
    )
    with pytest.raises(TypeError, match="Input value cannot be cast to Tensor."):
        polynomial("Hello world!")
    with pytest.raises(
        ValueError, match="Size of the input along the last axis should be 3, it is 5."
    ):
        polynomial(torch.zeros(3, 3, 5))


def test_get_wrong_tensor():
    polynomial = QuadraticPolynomial(quadratic, linear, constant, dtype=torch.float32)
    with pytest.raises(ValueError, match="Positive integer required."):
        polynomial[-1]
