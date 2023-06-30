from ..ising import Ising
from ..ising_interface import IsingInterface
from typing import List, Union
import torch
from numpy import argmin


class Binary(IsingInterface):

    """
    Quadratic Unconstrained Binary Optimization

    Given a matrix `Q` the value to minimize is the quadratic form
    `ΣΣ Q(i,j)b(i)b(j)` where the `b(i)`'s values are either `0` or `1`.

    This problem is a variant of an Ising model where
    the states vectors are binary values instead of spins.

    It can also be extended to a more general binary formulation of an
    Ising problem. Given a matrix `Q` (quadratic form), a vector `l`
    (linear form) and a constant `c`, the value to minimize is 
    `ΣΣ Q(i,j)b(i)b(j) + Σ l(i)b(i) + c` where the `b(i)`'s values
    are either `0` or `1`.
    """

    def __init__(self, matrix: torch.Tensor, vector: Union[torch.Tensor, None] = None, constant: Union[float, int, None] = None,
                dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(matrix, vector, constant, [0, 1], dtype, device)

    def to_ising(self) -> Ising:
        symmetrical_matrix = self.matrix + self.matrix.t()
        J = - .25 * symmetrical_matrix
        h = .5 * self.l + .25 * symmetrical_matrix @ torch.ones((len(self), 1), device=self.device)
        return Ising(J, h, self.dtype, self.device)

    def from_ising(self, ising: Ising) -> torch.Tensor:
        if ising.ground_state is not None:
            return .5 * (ising.ground_state + 1)


class QUBO(Binary):

    """
    Quadratic Unconstrained Binary Optimization

    Given a matrix `Q` the value to minimize is the quadratic form
    `ΣΣ Q(i,j)b(i)b(j)` where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(self, Q: torch.Tensor, dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(Q, None, None, dtype, device)
        self.Q = self.matrix
