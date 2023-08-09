import torch

from src.simulated_bifurcation.ising_core import IsingCore

J = torch.tensor(
    [
        [1, 2, 3],
        [2, 1, 4],
        [3, 4, 1],
    ],
    dtype=torch.float32,
)
h = torch.tensor([1, 0, -1], dtype=torch.float32)


def test_init_ising_model_from_tensors():
    ising = IsingCore(J, h)
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, h)
    assert ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3


def test_init_ising_model_from_arrays():
    ising = IsingCore(J.numpy(), h.numpy())
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, h)
    assert ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3


def test_ising_without_linear_term():
    ising = IsingCore(J)
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, torch.zeros(3))
    assert not ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3


def test_init_ising_with_null_h_vector():
    ising = IsingCore(J, torch.zeros(3))
    assert torch.equal(ising.J, J)
    assert torch.equal(ising.h, torch.zeros(3))
    assert not ising.linear_term
    assert len(ising) == 3
    assert ising.dimension == 3


def test_clip_vector_to_tensor():
    ising = IsingCore(J, h)
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
            dtype=torch.float32,
        ),
    )


def test_simulated_bifurcation_tensor():
    original = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=torch.float32,
    )
    ising = IsingCore(original)
    expected_result = torch.tensor(
        [
            [0, 3, 5],
            [3, 0, 7],
            [5, 7, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(ising.as_simulated_bifurcation_tensor(), expected_result)


def test_negative_ising():
    ising = IsingCore(J, h)
    negative_ising = -ising
    assert torch.equal(negative_ising.J, -J)
    assert torch.equal(negative_ising.h, -h)
    assert negative_ising.linear_term
    assert len(negative_ising) == 3
    assert negative_ising.dimension == 3
