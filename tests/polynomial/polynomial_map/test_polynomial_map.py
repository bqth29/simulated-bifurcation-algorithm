import numpy as np
import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.polynomial.polynomial_map import PolynomialMap
from src.simulated_bifurcation.polynomial.polynomial_map.errors import *

_map = {
    0: torch.tensor(2).to(dtype=torch.float32),
    1: torch.Tensor([1, 2, 3]).to(dtype=torch.float32),
    2: torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(dtype=torch.float32),
}


def test_init_polynomial_map():
    polynomial_map = PolynomialMap(_map)
    assert polynomial_map.dimension == 3
    assert polynomial_map.dtype == torch.float32
    assert torch.equal(torch.tensor(2), polynomial_map[0])
    assert torch.equal(torch.Tensor([1, 2, 3]), polynomial_map[1])
    assert torch.equal(
        torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), polynomial_map[2]
    )
    with pytest.raises(KeyError, match="3"):
        polynomial_map[3]


def test_init_polynomial_map_with_degree_0_only():
    polynomial_map = PolynomialMap({0: torch.tensor(2)})
    assert polynomial_map.dimension == 0


def test_add_tensor_to_polynomial_map():
    polynomial_map = PolynomialMap(_map)
    polynomial_map[3] = torch.arange(1, 28).reshape(3, 3, 3).to(dtype=torch.float32)
    assert torch.equal(torch.arange(1, 28).reshape(3, 3, 3), polynomial_map[3])
    assert isinstance(polynomial_map, PolynomialMap)
    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got 4 of type <class 'str'>.",
    ):
        polynomial_map["4"] = torch.arange(1, 82).reshape(3, 3, 3, 3)

    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got -1 of type <class 'int'>.",
    ):
        polynomial_map[-1] = torch.arange(1, 82).reshape(3, 3, 3, 3)
    with pytest.raises(
        PolynomialMapValueTypeError,
        match="Expected a tensor value type but got <class 'int'>.",
    ):
        polynomial_map[4] = 5
    with pytest.raises(
        PolynomialMapDegreeError,
        match="Wrong key usage. A 4 key was used to reference a 2-dimensional tensor.",
    ):
        polynomial_map[4] = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(
            dtype=torch.float32
        )
    with pytest.raises(
        PolynomialMapDataTypeError,
        match="Inconsistent dtype among map's tensors. Expected the dtype to be torch.float32 but got torch.int64.",
    ):
        polynomial_map[2] = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(
            dtype=torch.int64
        )
    with pytest.raises(
        PolynomialMapTensorDimensionError,
        match="All dimensions of the tensor must be equal.",
    ):
        polynomial_map[2] = torch.Tensor([[1, 2, 3], [4, 5, 6]]).to(dtype=torch.float32)
    with pytest.raises(
        PolynomialMapDimensionError,
        match="Inconsistent dimensions among map's tensors. Expected each dimension to be 3 but got 2.",
    ):
        polynomial_map[2] = torch.Tensor([[1, 2], [3, 4]]).to(dtype=torch.float32)


def test_init_polynomial_map_from_empty_dict_raises_error():
    with pytest.raises(
        EmptyPolynomialMapError,
        match="Cannot define a polynomiam map from an empty dictionnary.",
    ):
        PolynomialMap({})


def test_init_polynomial_map_from_non_dict_object_raises_error():
    with pytest.raises(
        PolynomialMapTypeError,
        match="A polynomial map must be a int -> tensor dictionnary.",
    ):
        PolynomialMap("Hello world!")


def test_init_polynomial_map_with_non_int_key_raises_error():
    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got 4 of type <class 'str'>.",
    ):
        PolynomialMap({"4": torch.arange(1, 82).reshape(3, 3, 3, 3)})


def test_init_polynomial_map_with_negative_int_key_raises_error():
    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got -1 of type <class 'int'>.",
    ):
        PolynomialMap({-1: torch.arange(1, 82).reshape(3, 3, 3, 3)})


def test_init_polynomial_map_non_tensor_value_raises_error():
    with pytest.raises(
        PolynomialMapValueTypeError,
        match="Expected a tensor value type but got <class 'str'>.",
    ):
        PolynomialMap({1: "Hello world!"})


def test_init_polynomial_map_with_inconsistency_between_key_and_tensor_dimension_raises_error():
    with pytest.raises(
        PolynomialMapDegreeError,
        match="Wrong key usage. A 1 key was used to reference a 2-dimensional tensor.",
    ):
        PolynomialMap({1: torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])})


def test_init_polynomial_map_with_inconsistent_dtype_raises_error():
    with pytest.raises(
        PolynomialMapDataTypeError,
        match="Inconsistent dtype among map's tensors. Expected the dtype to be torch.float32 but got torch.int32.",
    ):
        PolynomialMap(
            {
                1: torch.Tensor([1, 2, 3]).to(dtype=torch.float32),
                2: torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(
                    dtype=torch.int32
                ),
            }
        )


def test_init_polynomial_map_with_tensor_with_different_dimensions_raises_error():
    with pytest.raises(
        PolynomialMapTensorDimensionError,
        match="All dimensions of the tensor must be equal.",
    ):
        PolynomialMap({2: torch.Tensor([[1, 2, 3], [4, 5, 6]])})


def test_init_polynomial_map_with_inconsistent_dimension_raises_error():
    with pytest.raises(
        PolynomialMapDimensionError,
        match="Inconsistent dimensions among map's tensors. Expected each dimension to be 3 but got 2.",
    ):
        PolynomialMap(
            {
                1: torch.Tensor([1, 2, 3]),
                2: torch.Tensor([[1, 2], [3, 4]]),
            }
        )


def test_init_polynomial_map_from_expression():
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
    polynomial_map = PolynomialMap.from_expression(expression)
    assert_expected_polynomial_map(polynomial_map)


def test_init_polynomial_map_from_tensors():
    polynomial_map = PolynomialMap.from_tensors(
        2, torch.tensor([-1, -2, 1]), torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]])
    )
    assert_expected_polynomial_map(polynomial_map)
    polynomial_map = PolynomialMap.from_tensors(
        torch.tensor(2),
        torch.tensor([-1, -2, 1]),
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]]),
    )
    assert_expected_polynomial_map(polynomial_map)
    polynomial_map = PolynomialMap.from_tensors(
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]]), 2, np.array([-1, -2, 1])
    )
    assert_expected_polynomial_map(polynomial_map)
    with pytest.raises(
        TypeError, match="Expected tensors or arrays, got <class 'str'>."
    ):
        PolynomialMap.from_tensors(2, np.array([-1, -2, 1]), "Hello world!")
    with pytest.raises(
        ValueError,
        match="Declaring twice the same degree is ambiguous. Got two tensors for degree 2.",
    ):
        PolynomialMap.from_tensors(
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]]),
        )


def assert_expected_polynomial_map(polynomial_map: PolynomialMap):
    assert polynomial_map.dimension == 3
    assert polynomial_map.dtype == torch.float32
    assert torch.equal(torch.tensor(2), polynomial_map[0])
    assert torch.equal(torch.tensor([-1, -2, 1]), polynomial_map[1])
    assert torch.equal(
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]]), polynomial_map[2]
    )
