from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .abc_model import ABCModel


class _Status(Enum):
    FAILED = "failed"
    NOT_OPTIMIZED = "not optimized"
    SUCCESS = "success"


class Knapsack(ABCModel):
    domain = "binary"

    def __init__(
        self,
        weights: List[int],
        costs: List[Union[int, float]],
        max_weight: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.weights = weights[:]
        self.costs = costs[:]
        self.n_items = len(weights)
        self.max_weight = max_weight
        matrix = self.__make_matrix(dtype, device)
        vector = self.__make_vector(dtype, device)
        super().__init__(
            matrix, vector, float(self.__make_penalty()), dtype=dtype, device=device
        )

    @property
    def summary(self) -> Dict[str, Union[int, float, List[int]]]:
        """
        Displays the optimized knapsack's content and status
        as a dictionnary.

        Keys:

        - items : the list of items put in the knapsack (0 list if no object)
        - total_cost : the combined value of the objects in the knapsack (0 if no object)
        - total_weight : the combined weight of the objects in the knapsack (0 if no object)
        - status : the optimization status of the model (one of: success, failed or not optimized)
        """
        content = {
            "items": [],
            "total_cost": 0,
            "total_weight": 0,
            "status": _Status.NOT_OPTIMIZED.value,
        }
        if self.sb_result is not None:
            sb_result = self.sb_result[:, torch.argmin(self(self.sb_result.t())).item()]
            items = np.array(sb_result)[: self.n_items]
            weights_array = np.array(self.weights)
            costs_array = np.array(self.costs)
            content["items"] = np.arange(self.n_items)[items == 1].tolist()
            content["total_cost"] = np.sum(costs_array * items)
            content["total_weight"] = np.sum(weights_array * items)
            content["status"] = (
                _Status.SUCCESS
                if content["total_weight"] <= self.max_weight
                else _Status.FAILED
            ).value
        return content

    def __make_penalty(self) -> float:
        return np.sum(self.costs)

    def __make_matrix(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weights_array = np.array(self.weights).reshape(1, -1)
        range_array = np.arange(self.max_weight + 1).reshape(1, -1)
        matrix = np.block(
            [
                [weights_array.T @ weights_array, -weights_array.T @ range_array],
                [-range_array.T @ weights_array, 1 + range_array.T @ range_array],
            ]
        )
        return self.__make_penalty() * torch.tensor(matrix, dtype=dtype, device=device)

    def __make_vector(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        dim = self.n_items + self.max_weight + 1
        costs_array = np.array(self.costs)
        extended_cost_array = np.zeros(dim)
        extended_cost_array[: self.n_items] = costs_array
        extended_cost_array = extended_cost_array.reshape(-1, 1)
        unit_array = np.zeros(dim)
        unit_array[self.n_items :] = 1
        unit_array = unit_array.reshape(-1, 1)
        vector = -2 * self.__make_penalty() * unit_array - extended_cost_array
        return torch.tensor(vector, dtype=dtype, device=device).reshape(
            -1,
        )
