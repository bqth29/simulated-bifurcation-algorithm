from typing import Sequence, Union

from numpy import ndarray
from sympy import Poly
from torch import Tensor

__all__ = ["PolynomialLike", "TensorLike"]

TensorLike = Union[Tensor, ndarray, float, int]
PolynomialLike = Union[TensorLike, Sequence[TensorLike], Poly]
