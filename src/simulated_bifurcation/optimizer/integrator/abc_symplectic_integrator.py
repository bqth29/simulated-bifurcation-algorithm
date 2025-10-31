from abc import ABC

import torch

from ...core.tensor_bearer import TensorBearer


class ABCSymplecticIntegrator(ABC, TensorBearer):
    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__(dtype=dtype, device=device)
