from ..ising import Ising
from ..interface import IsingInterface
from typing import List, Union
import torch
from numpy import argmin


class Integer(IsingInterface):

    """
    Variant of an Ising model where the states vectors are positive integer values instead of spins.
    All these integer values are contained in the interval `[0; 2**N-1]` where `N` is an integer called
    the `number_of_bits`.

    Given a symmetric matrix `M`and a vector `v`, the value to minimize is 

    `-0.5 * ΣΣ M(i,j)e(i)e(j) + Σ v(i)e(i)`

    where the `e(i)`'s values are integer values of the range `[0; 2**N-1]`.
    """

    def __init__(self, matrix: torch.Tensor,
                vector: torch.Tensor,
                number_of_bits: int,
                dtype: torch.dtype=torch.float32,
                device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.matrix = matrix.to(dtype=dtype, device=device)
        self.dimension = matrix.shape[0]
        if vector is None:
            self.vector = torch.zeros((self.dimension, 1), device=device)
        else:
            self.vector = vector.reshape(-1, 1).to(dtype=dtype, device=device)
        self.number_of_bits = number_of_bits
        self.conversion_matrix = self.__conversion_matrix__()
        self.solution = None

    @property
    def objective_value(self) -> Union[float, None]: return self(self.solution)

    def __len__(self): return self.matrix.shape[0]

    def __call__(self, integer_vector: torch.Tensor) -> Union[None, float, List[float]]:

        if integer_vector is None: return None

        elif not isinstance(integer_vector, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(integer_vector)}.")

        elif torch.any(integer_vector != torch.round(integer_vector)):
            raise ValueError('Values must be integers.')
        
        elif torch.any(integer_vector > 2 ** self.number_of_bits - 1):
            raise ValueError(f'All values must be inferior to {2 ** self.number_of_bits - 1}.')

        elif integer_vector.shape in [(self.dimension,), (self.dimension, 1)]:
            integer_vector = integer_vector.reshape((-1, 1))
            M, v = self.matrix, self.vector.reshape((-1, 1))
            value = -.5 * integer_vector.t() @ M @ integer_vector + integer_vector.t() @ v
            return value.item()

        elif integer_vector.shape[0] == self.dimension:
            M, v = self.matrix, self.vector.reshape((-1, 1))
            values = torch.einsum('ij, ji -> i', integer_vector.t(), -.5 * M @ integer_vector + v)
            return values.tolist()

        else:
            raise ValueError(f"Expected {self.dimension} rows, got {integer_vector.shape[0]}.")
        
    def min(self, integer_vectors: torch.Tensor) -> torch.Tensor:

        """
        Returns the integer vector with the lowest objective value.
        """

        values = self(integer_vectors)
        best_value = argmin(values)
        return integer_vectors[:, best_value]
    
    def __conversion_matrix__(self) -> torch.Tensor:

        """
        Generates the integer-binary conversion matrix with the model's dimensions.
        Returns
        -------
        numpy.ndarray
        """  

        matrix = torch.zeros(
            (self.dimension * self.number_of_bits, self.dimension),
           device=self.device
        )

        for a in range(self.dimension):
            for b in range(self.number_of_bits):

                matrix[a*self.number_of_bits+b][a] = 2.0**b

        return matrix   

    def __to_Ising__(self) -> Ising:
        
        J = self.conversion_matrix @ self.matrix @ self.conversion_matrix.t()
        h = 2 * self.conversion_matrix @ self.vector \
            - self.conversion_matrix @ self.matrix @ self.conversion_matrix.t() \
            @ torch.ones((self.dimension * self.number_of_bits, 1), device=self.device)

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None:
        self.solution = .5 * self.conversion_matrix.t() @ (ising.ground_state + 1)
