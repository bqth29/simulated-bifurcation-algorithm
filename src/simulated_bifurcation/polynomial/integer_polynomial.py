from typing import Union

import numpy as np
import torch

from ..ising_core import IsingCore
from .ising_polynomial_interface import IsingPolynomialInterface


class IntegerPolynomial(IsingPolynomialInterface):

    """
    Given a matrix `Q` (quadratic form), a vector `l`
    (linear form) and a constant `c`, the value to minimize is
    `ΣΣ Q(i,j)n(i)n(j) + Σ l(i)n(i) + c` where the `n(i)`'s values
    are integers.
    """

    def __init__(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        vector: Union[torch.Tensor, np.ndarray, None] = None,
        constant: Union[float, int, None] = None,
        number_of_bits: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        if not isinstance(number_of_bits, int) or number_of_bits < 1:
            raise ValueError("The number of bits must be a non-negative integer.")
        super().__init__(
            matrix, vector, constant, [*range(2**number_of_bits)], dtype, device
        )
        self.number_of_bits = number_of_bits
        self.__int_to_bin_matrix = IntegerPolynomial.integer_to_binary_matrix(
            self.dimension, self.number_of_bits, self.device
        )

    @staticmethod
    def integer_to_binary_matrix(
        dimension: int, number_of_bits: int, device: str
    ) -> torch.Tensor:
        matrix = torch.zeros((dimension * number_of_bits, dimension), device=device)
        for row in range(dimension):
            for col in range(number_of_bits):
                matrix[row * number_of_bits + col][row] = 2.0**col
        return matrix

    def to_ising(self) -> IsingCore:
        symmetrical_matrix = 0.5 * (self.matrix + self.matrix.t())
        J = (
            -0.5
            * self.__int_to_bin_matrix
            @ symmetrical_matrix
            @ self.__int_to_bin_matrix.t()
        )
        h = (
            0.5 * self.__int_to_bin_matrix @ self.vector
            + 0.5
            * self.__int_to_bin_matrix
            @ self.matrix
            @ self.__int_to_bin_matrix.t()
            @ torch.ones((self.dimension * self.number_of_bits, 1), device=self.device)
        )
        return IsingCore(J, h, self.dtype, self.device)

    def convert_spins(self, ising: IsingCore) -> None:
        if ising.ground_state is not None:
            return 0.5 * self.__int_to_bin_matrix.t() @ (ising.ground_state + 1)
