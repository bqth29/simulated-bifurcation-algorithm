from typing import Union

import numpy as np
import torch

from ..polynomial import BinaryPolynomial


class QUBO(BinaryPolynomial):

    """
    Quadratic Unconstrained Binary Optimization

    Given a matrix `Q` the value to minimize is the quadratic form
    `ΣΣ Q(i,j)b(i)b(j)` where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(
        self,
        Q: Union[torch.Tensor, np.ndarray],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__(Q, None, None, dtype, device)
        self.Q = self.matrix
