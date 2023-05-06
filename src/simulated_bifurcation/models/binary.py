from ..ising import Ising
from ..interface import IsingInterface
from typing import List, Union
import torch
from numpy import argmin


class Binary(IsingInterface):

    """
    Variant of an Ising model where the states vectors are binary values instead of spins.
    Given a symmetric matrix `M`and a vector `v`, the value to minimize is 

    `-0.5 * ΣΣ M(i,j)b(i)b(j) + Σ v(i)b(i)`

    where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(self, matrix: torch.Tensor,
                vector: torch.Tensor,
                dtype: torch.dtype=torch.float32,
                device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.matrix = matrix.to(dtype=dtype, device=device)
        self.dimension = matrix.shape[0]
        if vector is None:
            self.vector = torch.zeros((self.dimension, 1), device=device)
        else:
            self.vector = vector.reshape(-1, 1).to(dtype=dtype, device=device)
        self.solution = None

    @property
    def objective_value(self) -> Union[float, None]: return self(self.solution)

    def __len__(self): return self.matrix.shape[0]

    def __call__(self, binary_vector: torch.Tensor) -> Union[None, float, List[float]]:

        if binary_vector is None: return None

        elif not isinstance(binary_vector, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(binary_vector)}.")

        elif torch.any(torch.abs(2 * binary_vector - 1) != 1):
            raise ValueError('Binary values must be either 0 or 1.')

        elif binary_vector.shape in [(self.dimension,), (self.dimension, 1)]:
            binary_vector = binary_vector.reshape((-1, 1))
            M, v = self.matrix, self.vector.reshape((-1, 1))
            value = -.5 * binary_vector.t() @ M @ binary_vector + binary_vector.t() @ v
            return value.item()

        elif binary_vector.shape[0] == self.dimension:
            M, v = self.matrix, self.vector.reshape((-1, 1))
            values = torch.einsum('ij, ji -> i', binary_vector.t(), -.5 * M @ binary_vector + v)
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

    def __to_Ising__(self) -> Ising:
        
        J = self.matrix
        h = 2 * self.vector - self.matrix @ torch.ones((len(self), 1), device=self.device)

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None:
        self.solution = .5 * (ising.ground_state + 1)
