from typing import Optional, Sequence, Union

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
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if len(_input) == 1 and isinstance(_input[0], Poly):
            self.__polynomial_map = PolynomialMap.from_expression(
                _input[0], dtype=dtype, device=device
            )
        else:
            self.__polynomial_map = PolynomialMap.from_tensors(
                *_input, dtype=dtype, device=device
            )

    @property
    def degree(self) -> int:
        return np.max(list(self.__polynomial_map.keys()))

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
        if degree in self.__polynomial_map.keys():
            return self.__polynomial_map[degree]
        if isinstance(degree, int):
            if degree >= 0:
                return torch.zeros(
                    (self.dimension,) * degree, dtype=self.dtype, device=self.device
                )
        raise ValueError("Positive integer required.")

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.__polynomial_map = PolynomialMap(
            {
                key: value.to(dtype=dtype, device=device)
                for key, value in self.__polynomial_map.items()
            }
        )
