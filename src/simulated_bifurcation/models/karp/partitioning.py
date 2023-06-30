from typing import Dict, List, Union
from ... import SpinPolynomial
from numpy import sum
import torch


class NumberPartioning(SpinPolynomial):

    """
    A solver that separates a set of numbers into two subsets whose 
    respective sums are as close as possible.
    """

    def __init__(self, numbers: list, dtype: torch.dtype=torch.float32,
        device: str = 'cpu') -> None:
        self.numbers = numbers
        tensor_numbers = torch.Tensor(self.numbers, device=device).reshape(-1, 1)
        super().__init__(-2 * tensor_numbers @ tensor_numbers.t(), None, None, dtype, device)

    @property
    def partition(self) -> Dict[str, Dict[str, Union[List[int], Union[int, None]]]]:
        result = {
            'left': {'values': [], 'sum': None},
            'right': {'values': [], 'sum': None}
        }
        if self.sb_result is None:
            return result
        left_subset = []
        right_subset = []
        for elt in range(len(self)):
            if self.sb_result[elt] > 0: left_subset.append(self.numbers[elt])
            else: right_subset.append(self.numbers[elt])
        result['left']['values'] = left_subset
        result['left']['sum'] = sum(left_subset)
        result['right']['values'] = right_subset
        result['right']['sum'] = sum(right_subset)
        return result        
