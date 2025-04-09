import pytest
import torch

from src.simulated_bifurcation.core.utils import safe_get_device, safe_get_dtype


def test_get_default_dtype():
    assert torch.float32 == safe_get_dtype(None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_get_proper_dtype(dtype: torch.dtype):
    assert dtype == safe_get_dtype(dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_get_unauthorized_dtype(dtype: torch.dtype):
    with pytest.raises(
        ValueError,
        match="The Simulated Bifurcation algorithm can only run with a torch.float32 or a torch.float64 dtype.",
    ):
        safe_get_dtype(dtype)


def test_get_default_device():
    assert torch.get_default_device() == safe_get_device(None)


def test_get_device():
    cpu = torch.device("cpu")
    assert cpu == safe_get_device(cpu)
    assert cpu == safe_get_device("cpu")
