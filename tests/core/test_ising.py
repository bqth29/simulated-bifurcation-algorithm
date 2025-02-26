import numpy as np
import pytest
import torch

from src.simulated_bifurcation.core import Ising

from ..test_utils import DEVICES, FLOAT_DTYPES, INT_DTYPES

J = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
h = [1, 0, -1]


@pytest.mark.parametrize(
    "use_tensor, use_linear_term, dtype, device",
    [
        (use_tensor, use_linear_term, dtype, device)
        for use_tensor in [True, False]
        for use_linear_term in [True, False]
        for dtype in [*FLOAT_DTYPES, None]
        for device in [*DEVICES, None]
    ],
)
def test_init_ising(
    use_tensor: bool, use_linear_term: bool, dtype: torch.dtype, device: torch.device
):
    input_J = torch.tensor(J, dtype=dtype, device=device) if use_tensor else np.array(J)
    input_h = (
        (torch.tensor(h, dtype=dtype, device=device) if use_tensor else np.array(h))
        if use_linear_term
        else None
    )
    ising = Ising(input_J, input_h, dtype, device)

    assert torch.equal(torch.tensor(J, dtype=dtype, device=device), ising._J)
    if use_linear_term:
        assert torch.equal(torch.tensor(h, dtype=dtype, device=device), ising._h)
    else:
        assert torch.equal(torch.zeros(3, dtype=dtype, device=device), ising._h)
    assert ising._has_linear_term == use_linear_term
    assert ising._dimension == 3
    assert ising._dtype == torch.float32 if dtype is None else dtype
    assert ising._device == torch.get_default_device() if device is None else device


@pytest.mark.parametrize(
    "dtype, device",
    [(dtype, device) for dtype in INT_DTYPES for device in DEVICES],
)
def test_init_ising_with_wrong_dtype(dtype: torch.dtype, device: torch.device):
    with pytest.raises(
        ValueError,
        match=f"Simulated Bifurcation optimization can only be carried out with torch.float32 or torch.float64 dtypes, but got {dtype}.",
    ):
        Ising(
            torch.tensor(J, dtype=torch.float32, device=device),
            dtype=dtype,
            device=device,
        )


def test_init_ising_with_non_2_dimensional_J():
    with pytest.raises(
        ValueError, match="Expected J to be 2-dimensional, but got 3 dimensions."
    ):
        Ising(torch.zeros(3, 3, 3))


def test_init_ising_with_non_square_J():
    with pytest.raises(
        ValueError, match="Expected J to be square, but got 3 rows and 2 columns."
    ):
        Ising(torch.zeros(3, 2))


def test_init_ising_with_inconsistant_h_shape():
    with pytest.raises(
        ValueError, match=r"Expected the shape of h to be 3, but got \(2,\)."
    ):
        Ising(torch.zeros(3, 3), torch.zeros(2))


@pytest.mark.parametrize(
    "use_linear_term, dtype, device",
    [
        (use_linear_term, dtype, device)
        for use_linear_term in [True, False]
        for dtype in FLOAT_DTYPES
        for device in DEVICES
    ],
)
def test_as_simulated_bifurcation_tensor(
    use_linear_term: bool, dtype: torch.dtype, device: torch.device
):
    ising = Ising(
        torch.tensor(J, dtype=dtype, device=device),
        torch.tensor(h, dtype=dtype, device=device) if use_linear_term else None,
        dtype,
        device,
    )

    if use_linear_term:
        expected_tensor = torch.tensor(
            [[0, 3, 5, -1], [3, 0, 7, 0], [5, 7, 0, 1], [-1, 0, 1, 0]],
            dtype=dtype,
            device=device,
        )
    else:
        expected_tensor = torch.tensor(
            [[0, 3, 5], [3, 0, 7], [5, 7, 0]], dtype=dtype, device=device
        )

    assert torch.equal(expected_tensor, ising.as_simulated_bifurcation_tensor())


@pytest.mark.parametrize(
    "use_linear_term, dtype, device",
    [
        (use_linear_term, dtype, device)
        for use_linear_term in [True, False]
        for dtype in FLOAT_DTYPES
        for device in DEVICES
    ],
)
def test_negative_ising(
    use_linear_term: bool, dtype: torch.dtype, device: torch.device
):
    ising = Ising(
        torch.tensor(J, dtype=dtype, device=device),
        torch.tensor(h, dtype=dtype, device=device) if use_linear_term else None,
        dtype,
        device,
    )
    negative_ising: Ising = -ising

    assert torch.equal(-torch.tensor(J, dtype=dtype, device=device), negative_ising._J)
    if use_linear_term:
        assert torch.equal(
            -torch.tensor(h, dtype=dtype, device=device), negative_ising._h
        )
    else:
        assert torch.equal(
            torch.zeros(3, dtype=dtype, device=device), negative_ising._h
        )
    assert negative_ising._has_linear_term == use_linear_term
    assert negative_ising._dimension == 3
    assert negative_ising._dtype == torch.float32 if dtype is None else dtype
    assert (
        negative_ising._device == torch.get_default_device()
        if device is None
        else device
    )
