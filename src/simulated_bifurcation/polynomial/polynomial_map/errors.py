from typing import Any, Type

import torch

__all__ = [
    "EmptyPolynomialMapError",
    "PolynomialMapDataTypeError",
    "PolynomialMapDegreeError",
    "PolynomialMapDimensionError",
    "PolynomialMapKeyTypeError",
    "PolynomialMapTensorDimensionError",
    "PolynomialMapTypeError",
    "PolynomialMapValueTypeError",
]


class EmptyPolynomialMapError(ValueError):
    def __init__(self) -> None:
        super().__init__("Cannot define a polynomiam map from an empty dictionnary.")


class PolynomialMapDataTypeError(ValueError):
    def __init__(self, expected_dtype: torch.dtype, actual_dtype: torch.dtype) -> None:
        super().__init__(
            "Inconsistent dtype among map's tensors. "
            f"Expected the dtype to be {expected_dtype} but got {actual_dtype}."
        )


class PolynomialMapDegreeError(ValueError):
    def __init__(self, key: int, ndim: int) -> None:
        super().__init__(
            "Wrong key usage. "
            f"A {key} key was used to reference a {ndim}-dimensional tensor."
        )


class PolynomialMapDimensionError(ValueError):
    def __init__(self, expected_dimension: int, actual_dimension: int) -> None:
        super().__init__(
            "Inconsistent dimensions among map's tensors. "
            "Expected each dimension to be "
            f"{expected_dimension} but got {actual_dimension}."
        )


class PolynomialMapTensorDimensionError(ValueError):
    def __init__(self) -> None:
        super().__init__("All dimensions of the tensor must be equal.")


class PolynomialMapKeyTypeError(ValueError):
    def __init__(self, key: Any) -> None:
        super().__init__(
            f"Expected a positive integer key type but got {key} of type {type(key)}."
        )


class PolynomialMapTypeError(TypeError):
    def __init__(self) -> None:
        super().__init__("A polynomial map must be a int -> tensor dictionnary.")


class PolynomialMapValueTypeError(ValueError):
    def __init__(self, value_type: Type) -> None:
        super().__init__(f"Expected a tensor value type but got {value_type}.")
