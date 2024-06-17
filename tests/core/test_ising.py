import pytest
import torch

from src.simulated_bifurcation.core import Ising

from ..utils import DEVICES, DTYPES

J = torch.tensor(
    [
        [1, 2, 3],
        [2, 1, 4],
        [3, 4, 1],
    ],
    dtype=torch.float32,
)
h = torch.tensor([1, 0, -1], dtype=torch.float32)


def init_J(dtype: torch.dtype, device: str):
    return torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=dtype,
        device=device,
    )


def init_h(dtype: torch.dtype, device: str):
    return torch.tensor([1, 0, -1], dtype=dtype, device=device)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_ising_model_from_tensors(dtype: torch.dtype, device: str):
    J = init_J(dtype, device)
    h = init_h(dtype, device)
    ising = Ising(J, h, dtype=dtype, device=device)
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, h)
    assert ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3
    assert ising.dtype == dtype
    assert ising.device == torch.device(device)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_ising_model_from_arrays(dtype: torch.dtype, device: str):
    J = init_J(dtype, device)
    h = init_h(dtype, device)
    ising = Ising(J.numpy(), h.numpy(), dtype=dtype, device=device)
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, h)
    assert ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3
    assert ising.dtype == dtype
    assert ising.device == torch.device(device)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_ising_without_linear_term(dtype: torch.dtype, device: str):
    J = init_J(dtype, device)
    ising = Ising(J, dtype=dtype, device=device)
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, torch.zeros(3))
    assert not ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3
    assert ising.dtype == dtype
    assert ising.device == torch.device(device)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_init_ising_with_null_h_vector(dtype: torch.dtype, device: str):
    J = init_J(dtype, device)
    ising = Ising(J, torch.zeros(3), dtype=dtype, device=device)
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, torch.zeros(3))
    assert not ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3
    assert ising.dtype == dtype
    assert ising.device == torch.device(device)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_clip_vector_to_tensor(dtype: torch.dtype, device: str):
    J = init_J(dtype, device)
    h = init_h(dtype, device)
    ising = Ising(J, h, dtype=dtype, device=device)
    attached = ising.clip_vector_to_tensor()
    assert torch.equal(
        attached,
        torch.tensor(
            [
                [1, 2, 3, -1],
                [2, 1, 4, 0],
                [3, 4, 1, 1],
                [-1, 0, 1, 0],
            ],
            dtype=dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_simulated_bifurcation_tensor(dtype: torch.dtype, device: str):
    original = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=dtype,
        device=device,
    )
    ising = Ising(original, dtype=dtype, device=device)
    expected_result = torch.tensor(
        [
            [0, 3, 5],
            [3, 0, 7],
            [5, 7, 0],
        ],
        dtype=dtype,
        device=device,
    )
    assert torch.equal(ising.as_simulated_bifurcation_tensor(), expected_result)


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in DTYPES for device in DEVICES],
)
def test_negative_ising(dtype: torch.dtype, device: str):
    J = init_J(dtype, device)
    h = init_h(dtype, device)
    ising = Ising(J, h, dtype=dtype, device=device)
    negative_ising = -ising
    assert torch.equal(negative_ising.J, -J)
    assert torch.equal(negative_ising.h, -h)
    assert negative_ising.linear_term
    assert len(negative_ising) == 3
    assert negative_ising.dimension == 3
    assert negative_ising.dtype == dtype
    assert negative_ising.device == torch.device(device)


def test_minimize():
    pass
