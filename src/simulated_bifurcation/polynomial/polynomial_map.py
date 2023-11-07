from typing import Dict, Sequence, Union

import numpy as np
import torch
from sympy import Poly

TensorLike = Union[torch.Tensor, np.ndarray]
PolynomialLike = Union[TensorLike, Sequence[TensorLike], Poly]
PolynomialMap = Dict[int, torch.Tensor]


class PolynomialError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class QuadraticPolynomialError(PolynomialError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Polynomial:
    def __init__(self, polynomial_map: PolynomialMap) -> None:
        # TODO: handle empty polynomial maps or constants
        self.polynomial_map = polynomial_map
        self.degree = np.max(polynomial_map.keys())
        self.dimension = self.polynomial_map[self.degree].shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.polynomial_map[self.degree].dtype

    @property
    def device(self) -> torch.device:
        return self.polynomial_map[self.degree].device

    def __call__(self):
        pass

    def __getitem__(self, degree: int) -> torch.Tensor:
        return (
            self.polynomial_map[degree]
            if degree in self.polynomial_map.keys()
            else torch.zeros(
                (self.dimension,) * degree, dtype=self.dtype, device=self.device
            )
        )


class QuadraticPolynomial(Polynomial):
    def __init__(self, polynomial_map: PolynomialMap) -> None:
        super().__init__(polynomial_map)
        if self.degree != 2:
            raise QuadraticPolynomialError(
                f"Expected a degree 2 polynomial, got {self.degree}."
            )


def polynomial_from_tensor(
    *tensors: Union[TensorLike, Sequence[TensorLike]],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> PolynomialMap:
    polynomial_map = {}
    dimension = None
    for tensor in tensors:
        if isinstance(tensor, (int, float)):
            degree = 0
        elif isinstance(tensor, (torch.Tensor, np.ndarray)):
            degree = tensor.ndim
            if degree >= 1:
                first_dimension = tensor.shape[0]
                if not np.all(np.array(tensor.shape) == first_dimension):
                    raise PolynomialError(
                        "All dimensions of a polynomial tensor must be equal."
                    )
                if dimension is None:
                    dimension = first_dimension
                elif dimension != first_dimension:
                    raise PolynomialError(
                        f"Inconsistent dimension accross tensors. Expected {dimension} but got {first_dimension}."
                    )
        else:
            raise TypeError(f"Expected tensors or arrays, got {type(tensor)}")
        if degree in polynomial_map.keys():
            polynomial_map[degree] += tensor.to(dtype=dtype, device=device)
        else:
            polynomial_map[degree] = tensor.to(dtype=dtype, device=device)
    return polynomial_map


def polynomial_from_expression(
    expression: Poly,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> PolynomialMap:
    polynomial_map = {}
    dimension = len(expression.gens)
    for coefficient, degrees in zip(expression.coeffs(), expression.monoms()):
        monomial_degree = np.sum(degrees)
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
    return polynomial_map
