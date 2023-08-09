from typing import Optional, Union

import numpy as np
import torch

from ..ising_core import IsingCore
from .ising_polynomial_interface import IsingPolynomialInterface


class IntegerPolynomial(IsingPolynomialInterface):

    """
    Order two multivariate polynomial that can be translated as an equivalent
    Ising problem to be solved with the Simulated Bifurcation algorithm.

    The polynomial is the combination of a quadratic and a linear form plus a
    constant term:

    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`

    where `Q` is a square matrix, `l` a vector a `c` a constant.

    The `n(i)`'s values must be integers.
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
        """
        Parameters
        ----------
        matrix : Tensor | ndarray
            the square matrix that manages the order-two terms in the
            polynomial (quadratic form matrix).
        vector : Tensor | ndarray | None, optional
            the vector that manages the order-one terms in the polynomial
            (linear form vector). `None` means no vector (default is `None`)
        constant : float | int | None, optional
            the constant term of the polynomial. `None` means no constant term
            (default is `None`)
        number_of_bits : int, optional
            the number of bits on which the input values are encoded (default
            is `1`)
        dtype : torch.dtype, optional
            the dtype used to encode polynomial's coefficients (default is
            `float32`)
        device : str, optional
            the device on which to perform the computations of the Simulated
            Bifurcation algorithm (default `"cpu"`)
        """
        if not isinstance(number_of_bits, int) or number_of_bits < 1:
            raise ValueError("The number of bits must be a non-negative integer.")
        super().__init__(
            matrix, vector, constant, [*range(2**number_of_bits)], dtype, device
        )
        self.number_of_bits = number_of_bits
        self.__int_to_bin_matrix = self.integer_to_binary_matrix(
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
            @ torch.ones(
                (self.dimension * self.number_of_bits),
                dtype=self.dtype,
                device=self.device,
            )
        )
        return IsingCore(J, h, self.dtype, self.device)

    def convert_spins(self, ising: IsingCore) -> Optional[torch.Tensor]:
        if ising.computed_spins is not None:
            return 0.5 * self.__int_to_bin_matrix.t() @ (ising.computed_spins + 1)
        return None
