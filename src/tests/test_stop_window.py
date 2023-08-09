import pytest
import torch

from src.simulated_bifurcation.optimizer import StopWindow

CONVERGENCE_THRESHOLD = 3
SPINS = 3
AGENTS = 2
SCENARIO = [
    torch.tensor(
        [
            [1, -1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, 1],
            [-1, -1],
            [1, 1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.float32,
    ),
]


def test_init_window():
    window = StopWindow(
        SPINS,
        AGENTS,
        CONVERGENCE_THRESHOLD,
        dtype=torch.float32,
        device="cpu",
        verbose=False,
    )
    assert window.n_spins == 3
    assert window.n_agents == 2
    assert window.convergence_threshold == 3
    assert window.shape == (3, 2)
    assert torch.equal(window.current_spins, torch.zeros((3, 2)))
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))


def test_wrong_convergence_threshold_value():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        StopWindow(
            SPINS, AGENTS, 30.0, dtype=torch.float32, device="cpu", verbose=False
        )
    with pytest.raises(ValueError):
        StopWindow(SPINS, AGENTS, 0, dtype=torch.float32, device="cpu", verbose=False)
    with pytest.raises(ValueError):
        StopWindow(SPINS, AGENTS, -42, dtype=torch.float32, device="cpu", verbose=False)
    with pytest.raises(ValueError):
        StopWindow(
            SPINS, AGENTS, 2**15, dtype=torch.float32, device="cpu", verbose=False
        )


def test_use_scenario():
    window = StopWindow(
        SPINS,
        AGENTS,
        CONVERGENCE_THRESHOLD,
        dtype=torch.float32,
        device="cpu",
        verbose=False,
    )

    # First update
    assert torch.equal(
        window.previously_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    window.update(SCENARIO[0])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(SCENARIO[0]), SCENARIO[0])
    assert torch.equal(window.current_spins, SCENARIO[0])
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.tensor([0, 0], dtype=torch.int16))
    assert torch.equal(
        window.newly_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.tensor([False, False], dtype=torch.bool))

    # Second update
    assert torch.equal(
        window.previously_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    window.update(SCENARIO[1])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(SCENARIO[1]), SCENARIO[1])
    assert torch.equal(window.current_spins, SCENARIO[1])
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.tensor([0, 0], dtype=torch.int16))
    assert torch.equal(
        window.newly_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.tensor([False, False], dtype=torch.bool))

    # Third update
    assert torch.equal(
        window.previously_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    window.update(SCENARIO[2])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(SCENARIO[2]), SCENARIO[2])
    assert torch.equal(window.current_spins, SCENARIO[2])
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.tensor([1, 0], dtype=torch.int16))
    assert torch.equal(
        window.newly_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.tensor([True, False], dtype=torch.bool))

    # Fourth update
    assert torch.equal(
        window.previously_bifurcated, torch.tensor([False, False], dtype=torch.bool)
    )
    window.update(SCENARIO[3])
    assert window.must_continue()
    assert window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(SCENARIO[3]), SCENARIO[3])
    assert torch.equal(window.current_spins, SCENARIO[3])
    assert torch.equal(
        window.final_spins,
        torch.tensor(
            [
                [-1, 0],
                [-1, 0],
                [1, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(window.stability, torch.tensor([2, 1], dtype=torch.float32))
    assert torch.equal(
        window.newly_bifurcated, torch.tensor([True, False], dtype=torch.bool)
    )
    assert torch.equal(window.bifurcated, torch.tensor([True, False], dtype=torch.bool))
    assert torch.equal(window.equal, torch.tensor([True, True], dtype=torch.bool))

    # Fourth update
    assert torch.equal(
        window.previously_bifurcated, torch.tensor([True, False], dtype=torch.bool)
    )
    window.update(SCENARIO[4])
    assert not window.must_continue()
    assert window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(SCENARIO[4]), SCENARIO[4])
    assert torch.equal(window.current_spins, SCENARIO[4])
    assert torch.equal(window.final_spins, SCENARIO[4])
    assert torch.equal(window.stability, torch.tensor([2, 2], dtype=torch.int16))
    assert torch.equal(
        window.newly_bifurcated, torch.tensor([False, True], dtype=torch.bool)
    )
    assert torch.equal(window.bifurcated, torch.tensor([True, True], dtype=torch.bool))
    assert torch.equal(window.equal, torch.tensor([True, True], dtype=torch.bool))
