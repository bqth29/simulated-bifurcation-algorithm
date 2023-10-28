import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.polynomial import (
    BinaryQuadraticPolynomial,
    IntegerQuadraticPolynomial,
    SpinQuadraticPolynomial,
)
from src.simulated_bifurcation.polynomial.builder import build_polynomial

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

    spin_polynomial = build_polynomial(expression, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    binary_polynomial = build_polynomial(expression, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    integer_polynomial = build_polynomial(expression, input_type="int3")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 3

    # Wrong input type

    with pytest.raises(ValueError):
        build_polynomial(expression, input_type="int2.5")

    with pytest.raises(ValueError):
        build_polynomial(expression, input_type="undefinedType")

    # Not quadratic polynomials

    with pytest.raises(ValueError, match="Expected degree 2 polynomial, got 1."):
        build_polynomial(poly(x + y + z))

    with pytest.raises(ValueError, match="Expected degree 2 polynomial, got 3."):
        build_polynomial(poly(x**3))


def test_build_polynomial_from_polynomial():
    base_spin_polynomial = SpinQuadraticPolynomial(quadratic, linear, constant)
    spin_polynomial = build_polynomial(base_spin_polynomial)
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    base_binary_polynomial = BinaryQuadraticPolynomial(quadratic, linear, constant)
    binary_polynomial = build_polynomial(base_binary_polynomial)
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    base_integer_polynomial = IntegerQuadraticPolynomial(quadratic, linear, constant, 4)
    integer_polynomial = build_polynomial(base_integer_polynomial)
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 4


def test_build_polynomial_from_tensor_tuple():
    # Spin polynomial

    spin_polynomial = build_polynomial(quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(torch.zeros(3), spin_polynomial.vector)
    assert torch.equal(torch.tensor(0), spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, linear, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(torch.tensor(0), spin_polynomial.constant)

    spin_polynomial = build_polynomial(linear, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(torch.tensor(0), spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, 2, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(torch.zeros(3), spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(2, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(torch.zeros(3), spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, constant, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(torch.zeros(3), spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(constant, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(torch.zeros(3), spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, linear, constant, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, linear, 2, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, constant, linear, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(quadratic, 2, linear, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(linear, quadratic, constant, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(linear, quadratic, 2, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(linear, constant, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(linear, 2, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(constant, linear, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(2, linear, quadratic, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(constant, quadratic, linear, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    spin_polynomial = build_polynomial(2, quadratic, linear, input_type="spin")
    assert isinstance(spin_polynomial, SpinQuadraticPolynomial)
    assert torch.equal(quadratic, spin_polynomial.matrix)
    assert torch.equal(linear, spin_polynomial.vector)
    assert torch.equal(constant, spin_polynomial.constant)

    with pytest.raises(
        ValueError,
        match="Defining two tensors for the same degree is ambiguous. Please merge them before to call this function.",
    ):
        build_polynomial(quadratic, quadratic, input_type="spin")

    with pytest.raises(
        ValueError, match="The polynomial must contain quadratic coefficients."
    ):
        build_polynomial(linear, input_type="spin")

    with pytest.raises(ValueError, match="Expected a tuple of size 1, 2 or 3; got 4."):
        build_polynomial(
            torch.zeros((3, 3, 3)), quadratic, linear, 2, input_type="spin"
        )

    with pytest.raises(ValueError, match="Expected all dimensions to be the same."):
        build_polynomial(torch.zeros((3, 2)), input_type="spin")

    # Binary polynomial

    binary_polynomial = build_polynomial(quadratic, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(torch.zeros(3), binary_polynomial.vector)
    assert torch.equal(torch.tensor(0), binary_polynomial.constant)

    binary_polynomial = build_polynomial(quadratic, linear, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(torch.tensor(0), binary_polynomial.constant)

    binary_polynomial = build_polynomial(linear, quadratic, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(torch.tensor(0), binary_polynomial.constant)

    binary_polynomial = build_polynomial(quadratic, 2, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(torch.zeros(3), binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(2, quadratic, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(torch.zeros(3), binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(quadratic, constant, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(torch.zeros(3), binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(constant, quadratic, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(torch.zeros(3), binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(
        quadratic, linear, constant, input_type="binary"
    )
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(quadratic, linear, 2, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(
        quadratic, constant, linear, input_type="binary"
    )
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(quadratic, 2, linear, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(
        linear, quadratic, constant, input_type="binary"
    )
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(linear, quadratic, 2, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(
        linear, constant, quadratic, input_type="binary"
    )
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(linear, 2, quadratic, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(
        constant, linear, quadratic, input_type="binary"
    )
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(2, linear, quadratic, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(
        constant, quadratic, linear, input_type="binary"
    )
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    binary_polynomial = build_polynomial(2, quadratic, linear, input_type="binary")
    assert isinstance(binary_polynomial, BinaryQuadraticPolynomial)
    assert torch.equal(quadratic, binary_polynomial.matrix)
    assert torch.equal(linear, binary_polynomial.vector)
    assert torch.equal(constant, binary_polynomial.constant)

    with pytest.raises(
        ValueError,
        match="Defining two tensors for the same degree is ambiguous. Please merge them before to call this function.",
    ):
        build_polynomial(quadratic, quadratic, input_type="binary")

    with pytest.raises(
        ValueError, match="The polynomial must contain quadratic coefficients."
    ):
        build_polynomial(linear, input_type="binary")

    with pytest.raises(ValueError, match="Expected a tuple of size 1, 2 or 3; got 4."):
        build_polynomial(
            torch.zeros((3, 3, 3)), quadratic, linear, 2, input_type="binary"
        )

    with pytest.raises(ValueError, match="Expected all dimensions to be the same."):
        build_polynomial(torch.zeros((3, 2)), input_type="binary")

    # Integer polynomial

    integer_polynomial = build_polynomial(quadratic, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(torch.zeros(3), integer_polynomial.vector)
    assert torch.equal(torch.tensor(0), integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(quadratic, linear, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(torch.tensor(0), integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(linear, quadratic, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(torch.tensor(0), integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(quadratic, 2, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(torch.zeros(3), integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(2, quadratic, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(torch.zeros(3), integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(quadratic, constant, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(torch.zeros(3), integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(constant, quadratic, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(torch.zeros(3), integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(
        quadratic, linear, constant, input_type="int2"
    )
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(quadratic, linear, 2, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(
        quadratic, constant, linear, input_type="int2"
    )
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(quadratic, 2, linear, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(
        linear, quadratic, constant, input_type="int2"
    )
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(linear, quadratic, 2, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(
        linear, constant, quadratic, input_type="int2"
    )
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(linear, 2, quadratic, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(
        constant, linear, quadratic, input_type="int2"
    )
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(2, linear, quadratic, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(
        constant, quadratic, linear, input_type="int2"
    )
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    integer_polynomial = build_polynomial(2, quadratic, linear, input_type="int2")
    assert isinstance(integer_polynomial, IntegerQuadraticPolynomial)
    assert torch.equal(quadratic, integer_polynomial.matrix)
    assert torch.equal(linear, integer_polynomial.vector)
    assert torch.equal(constant, integer_polynomial.constant)
    assert integer_polynomial.number_of_bits == 2

    with pytest.raises(
        ValueError,
        match="Defining two tensors for the same degree is ambiguous. Please merge them before to call this function.",
    ):
        build_polynomial(quadratic, quadratic, input_type="int2")

    with pytest.raises(
        ValueError, match="The polynomial must contain quadratic coefficients."
    ):
        build_polynomial(linear, input_type="int2")

    with pytest.raises(ValueError, match="Expected a tuple of size 1, 2 or 3; got 4."):
        build_polynomial(
            torch.zeros((3, 3, 3)), quadratic, linear, 2, input_type="int2"
        )

    with pytest.raises(ValueError, match="Expected all dimensions to be the same."):
        build_polynomial(torch.zeros((3, 2)), input_type="int2")


def test_build_polynomial_with_wrong_input_type():
    with pytest.raises(
        TypeError,
        match=r"Expected quadratic polynomial, a SymPy expression or a tuple of tensor\(s\)/array\(s\).",
    ):
        build_polynomial("Hello world!")
