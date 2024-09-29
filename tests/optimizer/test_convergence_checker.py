import pytest
import torch

from src.simulated_bifurcation.optimizer import ConvergenceChecker

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
        ConvergenceChecker(30.0, TENSOR, AGENTS, verbose=False)
    with pytest.raises(
        ValueError,
        match="convergence_threshold should be a positive integer, received 0.",
    ):
        ConvergenceChecker(0, TENSOR, AGENTS, verbose=False)
    with pytest.raises(
        ValueError,
        match="convergence_threshold should be a positive integer, received -42.",
    ):
        ConvergenceChecker(-42, TENSOR, AGENTS, verbose=False)
    with pytest.raises(
        ValueError,
        match="convergence_threshold should be less than or equal to 32767, received 32768.",
    ):
        ConvergenceChecker(2**15, TENSOR, AGENTS, verbose=False)


def test_use_scenario():
    """
    Ground state is degenerate: [-1, 1, -1] and [1, -1, 1]
    both reach the minimal energy value -6.

    Test of the convergence checker's behavior on 2 agents:
    - agent 1 converges to an optimal vector from step 3;
    - agent 2 oscillates in the optimal space from step 2.
    """
    convergence_checker = ConvergenceChecker(
        CONVERGENCE_THRESHOLD, TENSOR, AGENTS, verbose=False
    )

    #  Initial state
    assert torch.equal(convergence_checker.get_stored_spins(), torch.zeros(3, 2))
    assert torch.all(torch.isinf(convergence_checker.energies))
    assert torch.equal(convergence_checker.stability, torch.zeros(2))

    #  First update
    convergence_checker.update(SCENARIO[0])
    assert torch.equal(convergence_checker.energies, torch.tensor([2.0, 0.0]))
    assert torch.equal(convergence_checker.get_stored_spins(), torch.zeros((3, 2)))
    assert torch.equal(
        convergence_checker.stability, torch.tensor([0, 0], dtype=torch.int16)
    )

    #  Second update
    convergence_checker.update(SCENARIO[1])
    assert torch.equal(convergence_checker.energies, torch.tensor([0.0, -6.0]))
    assert torch.equal(convergence_checker.get_stored_spins(), torch.zeros((3, 2)))
    assert torch.equal(
        convergence_checker.stability, torch.tensor([0, 0], dtype=torch.int16)
    )

    #  Third update
    convergence_checker.update(SCENARIO[2])
    assert torch.equal(convergence_checker.energies, torch.tensor([-6.0, -6.0]))
    assert torch.equal(convergence_checker.get_stored_spins(), torch.zeros((3, 2)))
    assert torch.equal(
        convergence_checker.stability, torch.tensor([0, 1], dtype=torch.int16)
    )

    #  Fourth update
    convergence_checker.update(SCENARIO[3])
    assert torch.equal(convergence_checker.energies, torch.tensor([-6.0]))
    assert torch.equal(
        convergence_checker.get_stored_spins(),
        torch.tensor(
            [
                [0, -1],
                [0, 1],
                [0, -1],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        convergence_checker.stability, torch.tensor([1], dtype=torch.float32)
    )

    #  Fourth update
    convergence_checker.update(SCENARIO[4])
    assert torch.equal(convergence_checker.energies, torch.tensor([]))
    assert torch.equal(
        convergence_checker.get_stored_spins(),
        torch.tensor(
            [
                [-1, -1],
                [1, 1],
                [-1, -1],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(convergence_checker.stability, torch.tensor([]))
