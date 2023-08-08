from typing import Optional, Union

import numpy as np
import torch

from ..ising_core import IsingCore
from .ising_polynomial_interface import IsingPolynomialInterface


class BinaryPolynomial(IsingPolynomialInterface):

    """
    Order two multivariate polynomial that can be translated as an equivalent
    Ising problem to be solved with the Simulated Bifurcation algorithm.

    The polynomial is the combination of a quadratic and a linear form plus a
    constant term:

    `ΣΣ Q(i,j)b(i)b(j) + Σ l(i)b(i) + c`

    where `Q` is a square matrix, `l` a vector a `c` a constant.

    The `b(i)`'s values must be binary (either `0` or `1`).
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
        super().__init__(matrix, vector, constant, [0, 1], dtype, device)

    def to_ising(self) -> IsingCore:
        symmetrical_matrix = IsingCore.symmetrize(self.matrix)
        J = -0.5 * symmetrical_matrix
        h = 0.5 * self.vector + 0.5 * symmetrical_matrix @ torch.ones(
            len(self), dtype=self.dtype, device=self.device
        )
        return IsingCore(J, h, self.dtype, self.device)

    def convert_spins(self, ising: IsingCore) -> Optional[torch.Tensor]:
        if ising.computed_spins is not None:
            binary_vars = (ising.computed_spins + 1) / 2
        else:
            binary_vars = None
        return binary_vars
