from ..ising import Ising
from ..interface import IsingInterface
from typing import List, Union
import torch
from numpy import argmin


class QUBO(IsingInterface):

    """
    Quadratic Unconstrained Binary Optimization variant of an Ising model
    where the states vectors are binary values instead of spins.

    Given a matrix `Q` (quadratic form), a vector `l` (linear form) and a constant `c`,
    the value to minimize is 

    `ΣΣ Q(i,j)b(i)b(j) + Σ l(i)b(i) + c`

    where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(self, Q: torch.Tensor, l: Union[torch.Tensor, None], c: Union[float, int, None],
                dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.__dimension = Q.shape[0]
        self.__quadratic = Q.to(dtype=dtype, device=device)
        self.__linear = torch.zeros((self.dimension, 1), device=device) if l is None else l.reshape(-1, 1).to(dtype=dtype, device=device)
        self.__constant = 0 if c is None else c
        self.__best_binary_vector = None

    @property
    def Q(self) -> torch.Tensor:
        return self.__quadratic
    
    @property
    def l(self) -> torch.Tensor:
        return self.__linear
    
    @property
    def c(self) -> Union[float, int]:
        return self.__constant
    
    @property
    def dimension(self):
        return self.__dimension
    
    @property
    def best_binary_vector(self) -> Union[torch.Tensor, None]:
        return self.__best_binary_vector
    
    @property
    def best_objective_value(self) -> Union[float, None]:
        return self(self.best_binary_vector)

    def __len__(self):
        return self.dimension

    def __call__(self, binary_vector: torch.Tensor) -> Union[None, float, List[float]]:

        if binary_vector is None:
            return None

        elif not isinstance(binary_vector, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(binary_vector)}.")

        elif torch.any(torch.abs(2 * binary_vector - 1) != 1):
            raise ValueError('Binary values must be either 0 or 1.')

        elif binary_vector.shape in [(self.dimension,), (self.dimension, 1)]:
            binary_vector = binary_vector.reshape((-1, 1))
            value = binary_vector.t() @ self.Q @ binary_vector + self.l.t() @ binary_vector + self.c
            return value.item()

        elif binary_vector.shape[0] == self.dimension:
            values = torch.einsum('ij, ji -> i', binary_vector.t(), self.Q @ binary_vector + self.l) + self.c
            return values.tolist()

        else:
            raise ValueError(f"Expected {self.dimension} rows, got {binary_vector.shape[0]}.")
        
    def min(self, binary_vectors: torch.Tensor) -> torch.Tensor:

        """
        Returns the binary vector with the lowest objective value.
        """

        values = self(binary_vectors)
        best_value = argmin(values)
        return binary_vectors[:, best_value]

    def to_ising(self) -> Ising:
        symmetrical_Q = self.Q + self.Q.t()
        J = - .25 * symmetrical_Q
        h = .5 * self.l + .25 * symmetrical_Q @ torch.ones((len(self), 1), device=self.device)
        return Ising(J, h, self.dtype, self.device)

    def from_ising(self, ising: Ising) -> None:
        self.__best_binary_vector = .5 * (ising.ground_state + 1)
