from typing import Optional, Union

import numpy as np
import torch

from ..polynomial import BinaryQuadraticPolynomial


class QUBO(BinaryQuadraticPolynomial):

    """
    Quadratic Unconstrained Binary Optimization

    Given a matrix `Q` the value to minimize is the quadratic form
    `ΣΣ Q(i,j)b(i)b(j)` where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(
        self,
        Q: Union[torch.Tensor, np.ndarray],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(Q, None, None, dtype, device, silence_deprecation_warning=True)
        self.Q = self.matrix
