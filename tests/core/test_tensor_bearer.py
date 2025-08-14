import pytest
import torch

from src.simulated_bifurcation.core.tensor_bearer import TensorBearer

from ..test_utils import BOOLEANS, DEVICES, DTYPES

CPU = torch.device("cpu")


def test_init_with_default_dtype_and_device():
    tensor_bearer = TensorBearer()
    assert torch.float32 == tensor_bearer.dtype
    assert CPU == tensor_bearer.device


@pytest.mark.parametrize(
    "dtype, device, device_as_str",
    [
        (dtype, device, device_as_str)
        for dtype in DTYPES
        for device in DEVICES
        for device_as_str in BOOLEANS
    ],
)
def test_init_with_allowed_dtype_and_device(
    dtype: torch.dtype, device: torch.device, device_as_str: bool
):
    tensor_bearer = TensorBearer(
        dtype=dtype, device=str(device) if device_as_str else device
    )
    assert dtype == tensor_bearer.dtype
    assert device == tensor_bearer.device


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_init_with_unauthorized_dtype(dtype: torch.dtype):
    with pytest.raises(
        ValueError,
        match="The Simulated Bifurcation algorithm can only run with a torch.float32 or a torch.float64 dtype.",
    ):
        TensorBearer(dtype=dtype)
