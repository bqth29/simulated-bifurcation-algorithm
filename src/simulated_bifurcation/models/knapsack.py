from typing import List, Optional, Union

import numpy as np
import torch

from ..core.quadratic_polynomial import QuadraticPolynomial
from .abc_quadratic_model import ABCQuadraticModel


class Knapsack(ABCQuadraticModel):
    """
    Note
    ----
    It is advised to run `solve` in discrete mode with `torch.float32` dtype for an optimal behavior.
    """

    domain = "binary"
    sense = "minimize"

    def __init__(
        self, costs: List[Union[int, float]], weights: List[int], max_weight: int
    ):
        self.__costs = list(map(float, costs))
        self.__weights = list(map(float, weights))
        self.__max_weight = max_weight

    def _as_quadratic_polynomial(
        self, dtype: Optional[torch.dtype], device: Optional[Union[str, torch.device]]
    ) -> QuadraticPolynomial:
        # Define penalties for quadratic constraints
        weight_constraint_penalty = float(np.sum(self.__costs) + 1)
        only_one_weight_constraint_penalty = weight_constraint_penalty * (
            self.__max_weight + 1 + float(np.sum(self.__weights))
        )

        # Define arrays and tensors
        costs_array = np.array(self.__costs).reshape(-1, 1)
        weights_array = np.array(self.__weights).reshape(-1, 1)
        possible_total_weights_array = np.arange(self.__max_weight + 1).reshape(-1, 1)

        quadratic = np.block(
            [
                [
                    weight_constraint_penalty * weights_array @ weights_array.T,
                    -weight_constraint_penalty
                    * weights_array
                    @ possible_total_weights_array.T,
                ],
                [
                    -weight_constraint_penalty
                    * possible_total_weights_array
                    @ weights_array.T,
                    weight_constraint_penalty
                    * (possible_total_weights_array @ possible_total_weights_array.T)
                    + only_one_weight_constraint_penalty,
                ],
            ]
        )

        linear = np.concat(
            (
                -costs_array,
                -2
                * only_one_weight_constraint_penalty
                * np.ones(self.__max_weight + 1).reshape(-1, 1),
            )
        ).reshape(
            -1,
        )

        bias = only_one_weight_constraint_penalty

        return QuadraticPolynomial(quadratic, linear, bias, dtype=dtype, device=device)

    def _from_optimized_tensor(
        self, optimized_tensor: torch.Tensor, optimized_cost: torch.Tensor
    ) -> List[int]:
        return [
            index
            for index in range(len(self.__costs))
            if optimized_tensor[index].item() == 1.0
        ]


def knapsack(
    costs: List[Union[int, float]], weights: List[int], max_weight: int
) -> Knapsack:
    return Knapsack(costs, weights, max_weight)
