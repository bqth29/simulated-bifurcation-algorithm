from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from sympy import Poly

from .errors import *
from .types import *


class PolynomialMap(Dict[int, torch.Tensor]):
    def __init__(self, degree_to_tensor_map: Dict[int, torch.Tensor]) -> None:
        PolynomialMap.check_map(degree_to_tensor_map)
        super().__init__(degree_to_tensor_map)

    @property
    def dimension(self) -> int:
        for tensor in self.values():
            if tensor.ndim > 0:
                return tensor.shape[0]
        return 0

    @property
    def dtype(self) -> torch.dtype:
        for tensor in self.values():
            return tensor.dtype

    def __setitem__(self, __key: int, __value: torch.Tensor) -> None:
        PolynomialMap.__check_key(__key)
        PolynomialMap.__check_value_type(__value)
        PolynomialMap.__check_degree(__key, __value)
        PolynomialMap.__check_tensor_dtype(self.dtype, __value)
        PolynomialMap.__check_all_tensor_dimensions_equal(__value)
        PolynomialMap.__check_dimension_consistency(self.dimension, __value)
        return super().__setitem__(__key, __value)

    @staticmethod
    def check_map(_map: Dict[int, torch.Tensor]):
        dimension = None
        dtype = None
        PolynomialMap.__check_is_not_empty_dict(_map)
        for key in _map.keys():
            PolynomialMap.__check_key(key)
            tensor = _map[key]
            PolynomialMap.__check_value_type(tensor)
            PolynomialMap.__check_degree(key, tensor)
            if dtype is None:
                dtype = tensor.dtype
            else:
                PolynomialMap.__check_tensor_dtype(dtype, tensor)
            PolynomialMap.__check_all_tensor_dimensions_equal(tensor)
            if tensor.ndim > 0 and dimension is None:
                dimension = tensor.shape[0]
            else:
                PolynomialMap.__check_dimension_consistency(dimension, tensor)

    @staticmethod
    def __check_dimension_consistency(expected_dimension: int, tensor: torch.Tensor):
        if tensor.ndim > 0:
            if expected_dimension != tensor.shape[0]:
                raise PolynomialMapDimensionError(expected_dimension, tensor.shape[0])

    @staticmethod
    def __check_all_tensor_dimensions_equal(tensor: torch.Tensor):
        if tensor.ndim > 0:
            if not np.all(np.array(tensor.shape) == tensor.shape[0]):
                raise PolynomialMapTensorDimensionError()

    @staticmethod
    def __check_tensor_dtype(expected_dtype: torch.dtype, tensor: torch.Tensor):
        if expected_dtype != tensor.dtype:
            raise PolynomialMapDataTypeError(expected_dtype, tensor.dtype)

    @staticmethod
    def __check_degree(key: int, tensor: torch.Tensor):
        if key != tensor.ndim:
            raise PolynomialMapDegreeError(key, tensor.ndim)

    @staticmethod
    def __check_value_type(tensor: Any):
        if not isinstance(tensor, torch.Tensor):
            raise PolynomialMapValueTypeError(type(tensor))

    @staticmethod
    def __check_key(key: Any):
        if not isinstance(key, int):
            raise PolynomialMapKeyTypeError(key)
        if key < 0:
            raise PolynomialMapKeyTypeError(key)

    @staticmethod
    def __check_is_not_empty_dict(_map: Any):
        if not isinstance(_map, dict):
            raise PolynomialMapTypeError()
        if len(_map.keys()) == 0:
            raise EmptyPolynomialMapError()

    @classmethod
    def from_tensors(
        cls,
        *tensors: Union[TensorLike, Sequence[TensorLike]],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[int, torch.Tensor]:
        polynomial_map = {}
        for tensor in tensors:
            if isinstance(tensor, (int, float)):
                degree = 0
                tensor = torch.tensor(tensor, dtype=dtype, device=device)
            elif isinstance(tensor, (torch.Tensor, np.ndarray)):
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                tensor = tensor.to(dtype=dtype, device=device)
                degree = tensor.ndim
            else:
                raise TypeError(f"Expected tensors or arrays, got {type(tensor)}.")
            if degree in polynomial_map.keys():
                raise ValueError(
                    f"Declaring twice the same degree is ambiguous. Got two tensors for degree {degree}."
                )
            polynomial_map[degree] = tensor
        return cls(polynomial_map)

    @classmethod
    def from_expression(
        cls,
        expression: Poly,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[int, torch.Tensor]:
        polynomial_map = {}
        dimension = len(expression.gens)
        for coefficient, degrees in zip(expression.coeffs(), expression.monoms()):
            monomial_degree = int(np.sum(degrees))
            if monomial_degree not in polynomial_map.keys():
                polynomial_map[monomial_degree] = torch.zeros(
                    (dimension,) * monomial_degree, dtype=dtype, device=device
                )
            indices = []
            for index, multiplicity in enumerate(degrees):
                degree = multiplicity
                while degree > 0:
                    indices.append(index)
                    degree -= 1
            polynomial_map[monomial_degree][tuple(indices)] = float(coefficient)
        return cls(polynomial_map)
