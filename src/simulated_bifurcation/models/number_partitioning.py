from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..core.quadratic_polynomial import QuadraticPolynomial
from .abc_quadratic_model import ABCQuadraticModel


class NumberPartitioning(ABCQuadraticModel):
    """
    A solver that separates a set of numbers into two subsets, the
    respective sums of which are as close as possible.
    """

    domain = "spin"
    sense = "minimize"

    def __init__(self, numbers: List[Union[int, float]]) -> None:
        self.__numbers = numbers[:]

    def _as_quadratic_polynomial(
        self, dtype: Optional[torch.dtype], device: Optional[Union[str, torch.device]]
    ) -> QuadraticPolynomial:
        tensor_numbers = np.array(self.__numbers).reshape(-1, 1)
        return QuadraticPolynomial(
            tensor_numbers @ tensor_numbers.T, dtype=dtype, device=device
        )

    def _from_optimized_tensor(
        self, optimized_tensor: torch.Tensor, optimized_cost: torch.Tensor
    ) -> Dict:
        result = {
            "left": {"values": [], "sum": None},
            "right": {"values": [], "sum": None},
        }
        left_subset = []
        right_subset = []
        for elt in range(len(self.__numbers)):
            if optimized_tensor[elt].item() > 0:
                left_subset.append(self.__numbers[elt])
            else:
                right_subset.append(self.__numbers[elt])
        result["left"]["values"] = left_subset
        result["left"]["sum"] = float(np.sum(left_subset))
        result["right"]["values"] = right_subset
        result["right"]["sum"] = float(np.sum(right_subset))
        return result


def number_partitioning(numbers: List[Union[int, float]]) -> NumberPartitioning:
    return NumberPartitioning(numbers)
