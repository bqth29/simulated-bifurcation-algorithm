from typing import Optional, Union

import torch


class TensorBearer:
    """
    Utility abstract class to use as a parent class for objects relying on tensors.
    """

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.__safe_get_dtype(dtype)
        self.__safe_get_device(device)

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    @property
    def device(self) -> torch.device:
        return self.__device

    def __safe_get_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None:
            self.__dtype = torch.float32
            return
        elif dtype == torch.float32 or dtype == torch.float64:
            self.__dtype = dtype
            return
        raise ValueError(
            "The Simulated Bifurcation algorithm can only run with a torch.float32 or a torch.float64 dtype."
        )

    def __safe_get_device(self, device: Optional[Union[str, torch.device]]) -> None:
        self.__device = (
            torch.get_default_device() if device is None else torch.device(device)
        )
