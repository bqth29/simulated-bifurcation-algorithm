import pytest
import torch

from simulated_bifurcation.check_dtype import check_dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_check_float_dtype(dtype: torch.dtype):
    assert check_dtype(dtype) == dtype


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_check_int_or_float16_dtype(dtype: torch.dtype):
    with pytest.raises(
        ValueError,
        match="Only torch.float32 and torch.float64 are accepted for Simulated Bifurcation computations.",
    ):
        check_dtype(dtype)


def test_none_dtype():
    assert check_dtype(None) == torch.float32
