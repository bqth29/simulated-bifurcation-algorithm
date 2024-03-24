from typing import Optional

import torch


def check_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            "Only torch.float32 and torch.float64 are accepted for Simulated Bifurcation computations."
        )
    return dtype
