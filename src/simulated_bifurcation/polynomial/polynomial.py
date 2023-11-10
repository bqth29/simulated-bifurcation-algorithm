from typing import Sequence, Union

import numpy as np
import torch
from sympy import Poly

from .polynomial_map import PolynomialMap, TensorLike

PolynomialLike = Union[Sequence[TensorLike], Poly]


class Polynomial:
    def __init__(
        self,
        *_input: PolynomialLike,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        if len(_input) == 1 and isinstance(_input[0], Poly):
            self.__polynomial_map = PolynomialMap.from_expression(
                _input[0], dtype=dtype, device=device
            )
        else:
            self.__polynomial_map = PolynomialMap.from_tensors(
                _input, dtype=dtype, device=device
            )

    @property
    def degree(self) -> int:
        return np.max(self.__polynomial_map.keys())

    @property
    def dimension(self) -> int:
        return self.__polynomial_map.dimension

    @property
    def device(self) -> torch.device:
        return self.__polynomial_map.device

    @property
    def dtype(self) -> torch.dtype:
        return self.__polynomial_map.dtype

    def __getitem__(self, degree: int) -> torch.Tensor:
        return self.__polynomial_map[degree]

    def to(
        self,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        self.__polynomial_map = PolynomialMap(
            {
                key: value.to(dtype=dtype, device=device)
                for key, value in self.__polynomial_map
            }
        )
