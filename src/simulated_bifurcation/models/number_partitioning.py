from typing import Dict, List, Optional, Union

import torch
from numpy import sum

from .abc_model import ABCModel


class NumberPartitioning(ABCModel):
    """
    A solver that separates a set of numbers into two subsets, the
    respective sums of which are as close as possible.
    """

    domain = "spin"

    def __init__(
        self,
        numbers: list,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.numbers = numbers
        tensor_numbers = torch.tensor(self.numbers, dtype=dtype, device=device).reshape(
            -1, 1
        )
        super().__init__(
            tensor_numbers @ tensor_numbers.t(), dtype=dtype, device=device
        )

    @property
    def partition(self) -> Dict[str, Dict[str, Union[List[int], int, None]]]:
        result = {
            "left": {"values": [], "sum": None},
            "right": {"values": [], "sum": None},
        }
        if self.sb_result is None:
            return result

        i_min = torch.argmin(self(self.sb_result.t())).item()
        best_vector = self.sb_result[:, i_min]

        left_subset = []
        right_subset = []
        for elt in range(self._dimension):
            if best_vector[elt].item() > 0:
                left_subset.append(self.numbers[elt])
            else:
                right_subset.append(self.numbers[elt])
        result["left"]["values"] = left_subset
        result["left"]["sum"] = sum(left_subset)
        result["right"]["values"] = right_subset
        result["right"]["sum"] = sum(right_subset)
        return result
