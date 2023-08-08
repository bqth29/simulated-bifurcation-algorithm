from typing import Union

import numpy as np
import torch

from ..ising_core import IsingCore
from .ising_polynomial_interface import IsingPolynomialInterface


class SpinPolynomial(IsingPolynomialInterface):

    """
    Order two multivariate polynomial that can be translated as an equivalent
    Ising problem to be solved with the Simulated Bifurcation algorithm.

    The polynomial is the combination of a quadratic and a linear form plus a
    constant term:

    `ΣΣ Q(i,j)s(i)s(j) + Σ l(i)s(i) + c`

    where `Q` is a square matrix, `l` a vector a `c` a constant.

    The `s(i)`'s values must be spins (either `-1` or `1`).
    """

    def __init__(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        vector: Union[torch.Tensor, np.ndarray, None] = None,
        constant: Union[float, int, None] = None,
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
        dtype : torch.dtype, optional
            the dtype used to encode polynomial's coefficients (default is 
            `float32`)
        device : str, optional
            the device on which to perform the computations of the Simulated
            Bifurcation algorithm (default `"cpu"`)
        """
        super().__init__(matrix, vector, constant, [-1, 1], dtype, device)

    def to_ising(self) -> IsingCore:
        return IsingCore(-2 * self.matrix, self.vector, self.dtype, self.device)

    def convert_spins(self, ising: IsingCore) -> torch.Tensor:
        return ising.computed_spins
