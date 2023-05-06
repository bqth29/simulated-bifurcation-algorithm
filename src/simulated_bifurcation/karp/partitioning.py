from ..ising import Ising
from ..interface import IsingInterface
from numpy import sum
import torch


class NumberPartioning(IsingInterface):

    """
    A solver that separates a set of numbers into two subsets whose 
    respective sums are as close as possible.
    """

    def __init__(self, numbers: list, dtype: torch.dtype=torch.float32,
        device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.numbers = numbers
        self.partition = {
            'left': {'values': [], 'sum': None},
            'right': {'values': [], 'sum': None}
        }

    def __len__(self): return len(self.numbers)

    def __to_Ising__(self) -> Ising:
        
        tensor_numbers = torch.Tensor(self.numbers, device=self.device)
        J = -2 * tensor_numbers.reshape(-1, 1) @ tensor_numbers.reshape(1, -1)
        h = torch.zeros((len(self), 1))

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None:
        
        subset_left = []
        subset_right = []

        partition = ising.ground_state.reshape(-1,)

        for elt in range(len(self)):

            if partition[elt] > 0: subset_left.append(self.numbers[elt])
            else: subset_right.append(self.numbers[elt])

        self.partition['left']['values'] = subset_left
        self.partition['left']['sum'] = sum(subset_left)

        self.partition['right']['values'] = subset_right
        self.partition['right']['sum'] = sum(subset_right)
