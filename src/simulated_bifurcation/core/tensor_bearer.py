from typing import Optional, Union

import numpy as np
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
        self.__dtype = self.__safe_get_dtype(dtype)
        self.__device = self.__safe_get_device(device)

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    @property
    def device(self) -> torch.device:
        return self.__device

    @staticmethod
    def __safe_get_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
        if dtype is None:
            return torch.float32
        elif dtype == torch.float32 or dtype == torch.float64:
            return dtype
        raise ValueError(
            "The Simulated Bifurcation algorithm can only run with a torch.float32 or a torch.float64 dtype."
        )

    @staticmethod
    def __safe_get_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        return torch.get_default_device() if device is None else torch.device(device)

    def _safe_get_tensor(
        self, data: Union[torch.Tensor, np.ndarray, int, float]
    ) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return self._cast_tensor(data)
        elif isinstance(data, np.ndarray):
            return self._cast_tensor(torch.from_numpy(data))
        elif isinstance(data, (int, float)):
            return self._cast_tensor(torch.tensor(data))
        else:
            raise TypeError(
                "Tensors can only be interpreted from NumPy arrays or int/float values."
            )

    def _cast_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype, device=self.device)
