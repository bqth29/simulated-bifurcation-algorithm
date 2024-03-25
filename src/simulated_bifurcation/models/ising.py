from typing import Optional, Union

import numpy as np
import torch

from .abc_model import ABCModel


class Ising(ABCModel):
    """
    Implementation of the Ising model.

    Solving an Ising problem means searching the spin vector S (with values in
    {-1, 1}) such that, given a matrix J with zero diagonal and a
    vector h, the following quantity - called Ising energy - is minimal (S is
    then called the ground state): `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`
    """

    domain = "spin"

    def __init__(
        self,
        J: Union[torch.Tensor, np.ndarray],
        h: Union[torch.Tensor, np.ndarray, None] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        _tensors = (-0.5 * J,)
        if h is not None:
            _tensors += (h,)
        super().__init__(*_tensors, dtype=dtype, device=device)
        self.J = self[2]
        self.h = self[1]
