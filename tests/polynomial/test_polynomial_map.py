import numpy as np
import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.polynomial.polynomial_map import *

from ..utils import DEVICES, DTYPES


def init_map(dtype: torch.dtype, device: str):
    return {
        0: torch.tensor(2, dtype=dtype, device=device),
        1: torch.tensor([1, 2, 3], dtype=dtype, device=device),
        2: torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype, device=device),
    }


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map(dtype: torch.dtype, device: str):
    polynomial_map = PolynomialMap(init_map(dtype, device))
    assert polynomial_map.size == 3
    assert polynomial_map.dtype == dtype
    assert polynomial_map.device == torch.device(device)
    assert torch.equal(torch.tensor(2, dtype=dtype, device=device), polynomial_map[0])
    assert torch.equal(
        torch.tensor([1, 2, 3], dtype=dtype, device=device), polynomial_map[1]
    )
    assert torch.equal(
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype, device=device),
        polynomial_map[2],
    )
    with pytest.raises(KeyError, match="3"):
        polynomial_map[3]


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_with_degree_0_only(dtype: torch.dtype, device: str):
    polynomial_map = PolynomialMap({0: torch.tensor(2, dtype=dtype, device=device)})
    assert polynomial_map.size == 0


@pytest.mark.parametrize(
    "dtype1, dtype2, device",
    [
        (dtype1, dtype2, device)
        for dtype1 in DTYPES
        for dtype2 in DTYPES
        for device in DEVICES
        if dtype1 != dtype2
    ],
)
def test_add_tensor_to_polynomial_map(
    dtype1: torch.dtype, dtype2: torch.dtype, device: str
):
    polynomial_map = PolynomialMap(init_map(dtype1, device))
    polynomial_map[3] = torch.arange(1, 28, dtype=dtype1, device=device).reshape(
        3, 3, 3
    )
    assert torch.equal(
        torch.arange(1, 28, dtype=dtype1, device=device).reshape(3, 3, 3),
        polynomial_map[3],
    )
    assert isinstance(polynomial_map, PolynomialMap)
    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got 4 of type <class 'str'>.",
    ):
        polynomial_map["4"] = torch.arange(1, 82, dtype=dtype1, device=device).reshape(
            3, 3, 3, 3
        )

    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got -1 of type <class 'int'>.",
    ):
        polynomial_map[-1] = torch.arange(1, 82, dtype=dtype1, device=device).reshape(
            3, 3, 3, 3
        )
    with pytest.raises(
        PolynomialMapValueTypeError,
        match="Expected a tensor value type but got <class 'int'>.",
    ):
        polynomial_map[4] = 5
    with pytest.raises(
        PolynomialMapDegreeError,
        match="Wrong key usage. A 4 key was used to reference a 2-dimensional tensor.",
    ):
        polynomial_map[4] = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype1, device=device
        )
    with pytest.raises(
        PolynomialMapDataTypeError,
        match=f"Inconsistent dtype among map's tensors. Expected the dtype to be {dtype1} but got {dtype2}.",
    ):
        polynomial_map[2] = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype2, device=device
        )
    with pytest.raises(
        PolynomialMapTensorDimensionError,
        match="All dimensions of the tensor must be equal.",
    ):
        polynomial_map[2] = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=dtype1, device=device
        )
    with pytest.raises(
        PolynomialMapDimensionError,
        match="Inconsistent dimensions among map's tensors. Expected each dimension to be 3 but got 2.",
    ):
        polynomial_map[2] = torch.tensor([[1, 2], [3, 4]], dtype=dtype1, device=device)


def test_init_polynomial_map_from_empty_dict_raises_error():
    with pytest.raises(
        EmptyPolynomialMapError,
        match="Cannot define a polynomial map from an empty dictionnary.",
    ):
        PolynomialMap({})


def test_init_polynomial_map_from_non_dict_object_raises_error():
    with pytest.raises(
        PolynomialMapTypeError,
        match="A polynomial map must be a int -> tensor dictionnary.",
    ):
        PolynomialMap("Hello world!")


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_with_non_int_key_raises_error(
    dtype: torch.dtype, device: str
):
    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got 4 of type <class 'str'>.",
    ):
        PolynomialMap(
            {"4": torch.arange(1, 82, dtype=dtype, device=device).reshape(3, 3, 3, 3)}
        )


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_with_negative_int_key_raises_error(
    dtype: torch.dtype, device: str
):
    with pytest.raises(
        PolynomialMapKeyTypeError,
        match="Expected a positive integer key type but got -1 of type <class 'int'>.",
    ):
        PolynomialMap(
            {-1: torch.arange(1, 82, dtype=dtype, device=device).reshape(3, 3, 3, 3)}
        )


def test_init_polynomial_map_non_tensor_value_raises_error():
    with pytest.raises(
        PolynomialMapValueTypeError,
        match="Expected a tensor value type but got <class 'str'>.",
    ):
        PolynomialMap({1: "Hello world!"})


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_with_inconsistency_between_key_and_tensor_dimension_raises_error(
    dtype: torch.dtype, device: str
):
    with pytest.raises(
        PolynomialMapDegreeError,
        match="Wrong key usage. A 1 key was used to reference a 2-dimensional tensor.",
    ):
        PolynomialMap(
            {
                1: torch.tensor(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype, device=device
                )
            }
        )


@pytest.mark.parametrize(
    "dtype1, dtype2, device",
    [
        (dtype1, dtype2, device)
        for dtype1 in DTYPES
        for dtype2 in DTYPES
        for device in DEVICES
        if dtype1 != dtype2
    ],
)
def test_init_polynomial_map_with_inconsistent_dtype_raises_error(
    dtype1: torch.dtype, dtype2: torch.dtype, device: str
):
    with pytest.raises(
        PolynomialMapDataTypeError,
        match=f"Inconsistent dtype among map's tensors. Expected the dtype to be {dtype1} but got {dtype2}.",
    ):
        PolynomialMap(
            {
                1: torch.tensor([1, 2, 3], dtype=dtype1, device=device),
                2: torch.tensor(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype2, device=device
                ),
            }
        )


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_with_tensor_with_different_dimensions_raises_error(
    dtype: torch.dtype, device: str
):
    with pytest.raises(
        PolynomialMapTensorDimensionError,
        match="All dimensions of the tensor must be equal.",
    ):
        PolynomialMap(
            {2: torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)}
        )


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_with_inconsistent_dimension_raises_error(
    dtype: torch.dtype, device: str
):
    with pytest.raises(
        PolynomialMapDimensionError,
        match="Inconsistent dimensions among map's tensors. Expected each dimension to be 3 but got 2.",
    ):
        PolynomialMap(
            {
                1: torch.tensor([1, 2, 3], dtype=dtype, device=device),
                2: torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device),
            }
        )


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_from_expression(dtype: torch.dtype, device: str):
    x, y, z = symbols("x y z")
    expression = poly(
        x**2 + 2 * y**2 + 3 * z**2 - 2 * x * y - x * z - 3 * y * z - x - 2 * y + z + 2
    )
    polynomial_map = PolynomialMap.from_expression(
        expression, dtype=dtype, device=device
    )
    assert_expected_polynomial_map(polynomial_map, dtype, device)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_polynomial_map_from_tensors(dtype: torch.dtype, device: str):
    polynomial_map = PolynomialMap.from_tensors(
        2,
        torch.tensor([-1, -2, 1], dtype=dtype, device=device),
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
    )
    assert_expected_polynomial_map(polynomial_map, dtype, device)
    polynomial_map = PolynomialMap.from_tensors(
        torch.tensor(2, dtype=dtype, device=device),
        torch.tensor([-1, -2, 1], dtype=dtype, device=device),
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
    )
    assert_expected_polynomial_map(polynomial_map, dtype, device)
    polynomial_map = PolynomialMap.from_tensors(
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=dtype, device=device),
        2,
        np.array([-1, -2, 1]),
        dtype=dtype,
        device=device,
    )
    assert_expected_polynomial_map(polynomial_map, dtype, device)
    with pytest.raises(
        TypeError, match="Expected tensors or arrays, got <class 'str'>."
    ):
        PolynomialMap.from_tensors(2, np.array([-1, -2, 1]), "Hello world!")
    with pytest.raises(
        ValueError,
        match="Declaring twice the same degree is ambiguous. Got two tensors for degree 2.",
    ):
        PolynomialMap.from_tensors(
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype, device=device),
            torch.tensor(
                [[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=dtype, device=device
            ),
        )


def assert_expected_polynomial_map(
    polynomial_map: PolynomialMap, dtype: torch.dtype, device: str
):
    assert polynomial_map.size == 3
    assert polynomial_map.dtype == dtype
    assert polynomial_map.device == torch.device(device)
    assert torch.equal(torch.tensor(2, dtype=dtype, device=device), polynomial_map[0])
    assert torch.equal(
        torch.tensor([-1, -2, 1], dtype=dtype, device=device), polynomial_map[1]
    )
    assert torch.equal(
        torch.tensor([[1, -2, -1], [0, 2, -3], [0, 0, 3]], dtype=dtype, device=device),
        polynomial_map[2],
    )
