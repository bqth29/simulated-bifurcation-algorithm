from typing import Union

import numpy as np
import torch

from ..polynomial import SpinPolynomial


class Ising(SpinPolynomial):

    """
    Implementation of the Ising model.

    Solving an Ising problem means searching the spin vector S (with values in
    {-1, 1}) such that, given a matrix J with zero diagonal and a
    vector h, the following quantity - called Ising energy - is minimal (S is
    then called the ground state): `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`
    """

    def __init__(
        self,
        J: Union[torch.Tensor, np.ndarray],
        h: Union[torch.Tensor, np.ndarray, None] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__(-0.5 * J, h, None, dtype, device)
        self.J = J
        self.h = h
