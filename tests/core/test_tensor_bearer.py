import pytest
import torch
from numpy import array

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


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_convert_data_to_tensor(dtype: torch.dtype, device: torch.device):
    tensor_bearer = TensorBearer(dtype=dtype, device=device)
    data = [[1.0, 2.0], [3.0, 4.0]]
    expected_data_tensor = torch.tensor(data, dtype=dtype, device=device)
    expected_number_tensor = torch.tensor(2.0, dtype=dtype, device=device)

    # From tensor
    assert torch.equal(
        expected_data_tensor, tensor_bearer._safe_get_tensor(expected_data_tensor)
    )

    # From NumPy array
    assert torch.equal(
        expected_data_tensor, tensor_bearer._safe_get_tensor(array(data))
    )

    # From single numeric value
    assert torch.equal(expected_number_tensor, tensor_bearer._safe_get_tensor(2))
    assert torch.equal(expected_number_tensor, tensor_bearer._safe_get_tensor(2.0))

    # From "raw" data
    with pytest.raises(
        TypeError,
        match="Tensors can only be interpreted from NumPy arrays or int/float values.",
    ):
        # noinspection PyTypeChecker
        tensor_bearer._safe_get_tensor(data)


@pytest.mark.parametrize(
    "from_dtype, to_dtype, device",
    [
        (from_dtype, to_dtype, device)
        for from_dtype in DTYPES
        for to_dtype in DTYPES
        for device in DEVICES
    ],
)
def test_cast_tensor(
    from_dtype: torch.dtype, to_dtype: torch.dtype, device: torch.device
):
    tensor_bearer = TensorBearer(dtype=to_dtype, device=device)
    data = [[1.0, 2.0], [3.0, 4.0]]
    assert torch.equal(
        torch.tensor(data, dtype=to_dtype, device=device),
        tensor_bearer._cast_tensor(torch.tensor(data, dtype=from_dtype, device=device)),
    )
