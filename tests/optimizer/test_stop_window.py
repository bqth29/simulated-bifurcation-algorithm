import pytest
import torch

from src.simulated_bifurcation.optimizer import StopWindow

TENSOR = torch.tensor([[1.0, 0.5, -1.0], [0.5, 0.0, 1.0], [-1.0, 1.0, -2.0]])
CONVERGENCE_THRESHOLD = 3
SPINS = 3
AGENTS = 2
SCENARIO = [
    torch.tensor(
        [
            [-1, -1],
            [1, -1],
            [1, -1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, -1],
            [-1, 1],
            [1, -1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, 1],
            [1, -1],
            [-1, 1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, -1],
            [1, 1],
            [-1, -1],
        ],
        dtype=torch.float32,
    ),
    # 1 agents has converged and was removed from the oscillators
    torch.tensor(
        [
            [-1],
            [1],
            [-1],
        ],
        dtype=torch.float32,
    ),
]


def test_wrong_convergence_threshold_value():
    with pytest.raises(
        TypeError, match="convergence_threshold should be an integer, received 30.0."
    ):
        # noinspection PyTypeChecker
        StopWindow(30.0, TENSOR, AGENTS, verbose=False)
    with pytest.raises(
        ValueError,
        match="convergence_threshold should be a positive integer, received 0.",
    ):
        StopWindow(0, TENSOR, AGENTS, verbose=False)
    with pytest.raises(
        ValueError,
        match="convergence_threshold should be a positive integer, received -42.",
    ):
        StopWindow(-42, TENSOR, AGENTS, verbose=False)
    with pytest.raises(
        ValueError,
        match="convergence_threshold should be less than or equal to 32767, received 32768.",
    ):
        StopWindow(2**15, TENSOR, AGENTS, verbose=False)


def test_use_scenario():
    """
    Ground state is degenerate: [-1, 1, -1] and [1, -1, 1]
    both reach the minimal energy value -6.

    Test of the stop window's behavior on 2 agents:
    - agent 1 converges to an optimal vector from step 3;
    - agent 2 oscillates in the optimal space from step 2.
    """
    window = StopWindow(CONVERGENCE_THRESHOLD, TENSOR, AGENTS, verbose=False)

    #  Initial state
    assert torch.equal(window.get_stored_spins(), torch.zeros(3, 2))
    assert torch.all(torch.isinf(window.energies))
    assert torch.equal(window.stability, torch.zeros(2))

    #  First update
    window.update(SCENARIO[0])
    assert torch.equal(window.energies, torch.tensor([2.0, 0.0]))
    assert torch.equal(window.get_stored_spins(), torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.tensor([0, 0], dtype=torch.int16))

    #  Second update
    window.update(SCENARIO[1])
    assert torch.equal(window.energies, torch.tensor([0.0, -6.0]))
    assert torch.equal(window.get_stored_spins(), torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.tensor([0, 0], dtype=torch.int16))

    #  Third update
    window.update(SCENARIO[2])
    assert torch.equal(window.energies, torch.tensor([-6.0, -6.0]))
    assert torch.equal(window.get_stored_spins(), torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.tensor([0, 1], dtype=torch.int16))

    #  Fourth update
    window.update(SCENARIO[3])
    assert torch.equal(window.energies, torch.tensor([-6.0]))
    assert torch.equal(
        window.get_stored_spins(),
        torch.tensor(
            [
                [0, -1],
                [0, 1],
                [0, -1],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(window.stability, torch.tensor([1], dtype=torch.float32))

    #  Fourth update
    window.update(SCENARIO[4])
    assert torch.equal(window.energies, torch.tensor([]))
    assert torch.equal(
        window.get_stored_spins(),
        torch.tensor(
            [
                [-1, -1],
                [1, 1],
                [-1, -1],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(window.stability, torch.tensor([]))
