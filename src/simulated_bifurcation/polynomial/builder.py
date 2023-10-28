import re
from collections import defaultdict
from typing import Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from sympy import Poly

from .base_multivariate_polynomial import BaseMultivariateQuadraticPolynomial
from .binary_polynomial import BinaryQuadraticPolynomial
from .integer_polynomial import IntegerQuadraticPolynomial
from .spin_polynomial import SpinQuadraticPolynomial

TensorLike = Union[torch.Tensor, np.ndarray]
Number = Union[float, int]
PolynomialLike = TypeVar(
    "PolynomialLike",
    BaseMultivariateQuadraticPolynomial,
    Poly,
    TensorLike,
    Tuple[TensorLike, TensorLike],
    Tuple[TensorLike, Number],
    Tuple[Number, TensorLike],
    Tuple[TensorLike, TensorLike, TensorLike],
    Tuple[TensorLike, TensorLike, Number],
    Tuple[TensorLike, Number, TensorLike],
    Tuple[Number, TensorLike, TensorLike],
)

INTEGER_POLYNOMIAL_REGEX = re.compile("^int[1-9][0-9]*$")


def build_polynomial(
    *_input: PolynomialLike,
    input_type: Optional[str] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[Union[str, torch.device]] = "cpu",
) -> BaseMultivariateQuadraticPolynomial:
    polynomial_type, number_of_bits = __get_polynomial_type(
        "spin" if input_type is None else input_type
    )
    if len(_input) == 1:
        _input_polynomial = _input[0]
        if isinstance(_input_polynomial, BaseMultivariateQuadraticPolynomial):
            _input_polynomial.to(dtype=dtype, device=device)
            return _input_polynomial
        if isinstance(_input_polynomial, Poly):
            return __build_polynomial_from_expression(
                _input_polynomial, polynomial_type, number_of_bits, dtype, device
            )
    return __build_polynomial_from_tensor_tuple(
        _input, polynomial_type, number_of_bits, dtype, device
    )


def __get_polynomial_type(
    input_type: str,
) -> Union[Type[BaseMultivariateQuadraticPolynomial], Optional[int]]:
    if input_type == "spin":
        return SpinQuadraticPolynomial, None
    if input_type == "binary":
        return BinaryQuadraticPolynomial, None
    if INTEGER_POLYNOMIAL_REGEX.match(input_type) is None:
        raise ValueError(
            f'Input type must be one of "spin" or "binary", or be a string starting'
            f'with "int" and be followed by a positive integer.\n'
            f"More formally, it should match the following regular expression.\n"
            f"{INTEGER_POLYNOMIAL_REGEX}\n"
            f'Examples: "int7", "int42", ...'
        )
    number_of_bits = int(input_type[3:])
    return IntegerQuadraticPolynomial, number_of_bits


def __build_polynomial_from_expression(
    _input: Poly,
    polynomial_type: Type[BaseMultivariateQuadraticPolynomial],
    number_of_bits: int,
    dtype: Optional[torch.dtype],
    device: Optional[Union[str, torch.device]],
) -> BaseMultivariateQuadraticPolynomial:
    if polynomial_type == IntegerQuadraticPolynomial:
        return IntegerQuadraticPolynomial.from_expression(
            expression=_input,
            number_of_bits=number_of_bits,
            dtype=dtype,
            device=device,
        )
    return polynomial_type.from_expression(
        expression=_input,
        dtype=dtype,
        device=device,
    )


def __build_polynomial_from_tensor_tuple(
    _input: Tuple,
    polynomial_type: Type[BaseMultivariateQuadraticPolynomial],
    number_of_bits: int,
    dtype: Optional[torch.dtype],
    device: Optional[Union[str, torch.device]],
) -> BaseMultivariateQuadraticPolynomial:
    _input_length = len(_input)
    if _input_length not in [1, 2, 3]:
        raise ValueError(f"Expected a tuple of size 1, 2 or 3; got {_input_length}.")
    degree_to_index_map = defaultdict(lambda: None)
    for index, tensor in enumerate(_input):
        degree = __get_tensor_degree(tensor)
        if degree in degree_to_index_map.keys():
            raise ValueError(
                "Defining two tensors for the same degree is ambiguous. "
                "Please merge them before to call this function."
            )
        else:
            degree_to_index_map[degree] = index
    if 2 not in degree_to_index_map.keys():
        raise ValueError("The polynomial must contain quadratic coefficients.")
    matrix = _input[degree_to_index_map[2]]
    vector = None if degree_to_index_map[1] is None else _input[degree_to_index_map[1]]
    constant = (
        None if degree_to_index_map[0] is None else _input[degree_to_index_map[0]]
    )
    if polynomial_type == IntegerQuadraticPolynomial:
        return IntegerQuadraticPolynomial(
            matrix=matrix,
            vector=vector,
            constant=constant,
            number_of_bits=number_of_bits,
            dtype=dtype,
            device=device,
        )
    else:
        return polynomial_type(
            matrix=matrix,
            vector=vector,
            constant=constant,
            dtype=dtype,
            device=device,
        )


def __get_tensor_degree(tensor: Union[Number, TensorLike]) -> int:
    if isinstance(tensor, (int, float)) or (
        isinstance(tensor, (torch.Tensor, np.ndarray)) and tensor.ndim == 0
    ):
        return 0
    if isinstance(tensor, (torch.Tensor, np.ndarray)):
        ndim = tensor.ndim
        shape = tensor.shape
        if ndim == 1 or (ndim == 2 and (shape[0] == 1 or shape[1] == 1)):
            return 1
        else:
            first_dim = shape[0]
            if np.all(np.array(shape) == first_dim):
                return ndim
            raise ValueError("Expected all dimensions to be the same.")
    raise TypeError(
        "Expected quadratic polynomial, a SymPy expression or a tuple of tensor(s)/array(s)."
    )
