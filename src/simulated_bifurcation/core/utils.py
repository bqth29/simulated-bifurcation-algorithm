from typing import Optional, Union

import torch


def safe_get_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    elif dtype == torch.float32 or dtype == torch.float64:
        return dtype
    raise ValueError(
        "The Simulated Bifurcation algorithm can only run with a torch.float32 or a torch.float64 dtype."
    )


def safe_get_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    return torch.get_default_device() if device is None else torch.device(device)
