from typing import Dict, List, Union

import numpy as np
import torch

from ..polynomial import BinaryPolynomial


class Knapsack(BinaryPolynomial):
    def __init__(
        self,
        weights: List[int],
        costs: List[Union[int, float]],
        max_weight: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.weights = weights[:]
        self.costs = costs[:]
        self.n_items = len(weights)
        self.max_weight = max_weight
        matrix = self.__make_matrix(dtype, device)
        vector = self.__make_vector(dtype, device)
        super().__init__(matrix, vector, None, dtype, device)

    @property
    def content(self) -> Dict[str, Union[int, float, List[int]]]:
        content = {"items": [], "total_cost": 0, "total_weight": 0}
        if self.sb_result is not None:
            sb_result = self.sb_result[
                :, torch.argmin(self(self.sb_result.t())).item()
            ]
            items = np.array(sb_result)[: self.n_items]
            weights_array = np.array(self.weights)
            costs_array = np.array(self.costs)
            content["items"] = np.arange(1, self.n_items + 1)[items == 1].tolist()
            content["total_cost"] = np.sum(costs_array * items)
            content["total_weight"] = np.sum(weights_array * items)
        return content

    def __make_matrix(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weights_array = np.array(self.weights).reshape(1, -1)
        range_array = np.arange(self.max_weight + 1).reshape(1, -1)
        matrix = np.block(
            [
                [weights_array.T @ weights_array, -2 * weights_array.T @ range_array],
                [-2 * range_array.T @ weights_array, 1 + range_array.T @ range_array],
            ]
        )
        return torch.tensor(matrix, dtype=dtype, device=device)

    def __make_vector(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        dim = self.n_items + self.max_weight + 1
        costs_array = np.array(self.costs)
        extended_cost_array = np.zeros(dim)
        extended_cost_array[: self.n_items] = costs_array
        extended_cost_array = extended_cost_array.reshape(-1, 1)
        unit_array = np.zeros(dim)
        unit_array[self.n_items :] = 1
        unit_array = unit_array.reshape(-1, 1)
        penalty = 0.5 / costs_array.max()
        vector = -2 * unit_array - penalty * extended_cost_array
        return torch.tensor(vector, dtype=dtype, device=device)
