from typing import Union

from numpy import ndarray
from torch import Tensor, tensor


def cast_to_tensor(_input: Union[Tensor, ndarray]) -> Tensor:
    if isinstance(_input, Tensor):
        return _input
    if isinstance(_input, ndarray):
        return tensor(_input)
    raise TypeError(f"Expected tensor or array, got {_input.__class__.__name__}.")
