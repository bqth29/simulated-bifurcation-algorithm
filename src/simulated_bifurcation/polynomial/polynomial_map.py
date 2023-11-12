from typing import Any, Dict, Sequence, Type, Union

import numpy as np
import torch
from sympy import Poly

TensorLike = Union[torch.Tensor, np.ndarray, float, int]


class EmptyPolynomialMapError(ValueError):
    def __init__(self) -> None:
        super().__init__("Cannot define a polynomial map from an empty dictionnary.")


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


class PolynomialMap(Dict[int, torch.Tensor]):
    """
    Utility class to provide a data structure that properly defined multivariate
    polynomials of any dimension.

    A multivariate polynomial is a sum of homogeneous polynomial (hyperlinear forms),
    i.e. a sequence of multi-dimensional tensors with the same size. The degree of each
    homogeneous polynomial is the number of dimensions of its coeffcients tensor.

    A polynomial map is a dictionnary that maps a degree (positive integer) to a
    coefficient tensors with as many dimensions as degree, all with the same size.

    Parameters
    ----------
    _map : Dict[int, Tensor]
        The degree to tensor map that defines a multivariate polynomial.
    """

    def __init__(self, degree_to_tensor_map: Dict[int, torch.Tensor]) -> None:
        PolynomialMap.check_map(degree_to_tensor_map)
        super().__init__(degree_to_tensor_map)

    @property
    def size(self) -> int:
        """
        Common size of all the tensors in the map.

        Returns
        -------
        int

        """
        for tensor in self.values():
            if tensor.ndim > 0:
                return tensor.shape[0]
        return 0

    @property
    def device(self) -> torch.device:
        """
        Device on which of all the tensors in the map are defined.

        Returns
        -------
        torch.device

        """
        for tensor in self.values():
            return tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Common data type of all the tensors in the map.

        Returns
        -------
        torch.dtype

        """
        for tensor in self.values():
            return tensor.dtype

    def __setitem__(self, __key: int, __value: torch.Tensor) -> None:
        PolynomialMap.__check_key(__key)
        PolynomialMap.__check_value_type(__value)
        PolynomialMap.__check_degree(__key, __value)
        PolynomialMap.__check_tensor_dtype(self.dtype, __value)
        PolynomialMap.__check_all_tensor_dimensions_equal(__value)
        PolynomialMap.__check_dimension_consistency(self.size, __value)
        return super().__setitem__(__key, __value)

    @staticmethod
    def check_map(_map: Dict[int, torch.Tensor]):
        """
        Checks that the polynomial map is correctly defined.

        Parameters
        ----------
        _map : Dict[int, torch.Tensor]
            The polynomial map to assess.
        """
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
