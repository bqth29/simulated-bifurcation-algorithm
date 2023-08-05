import torch

from src.simulated_bifurcation.optimizer import StopWindow

CONVERGENCE_THRESHOLD = 3
SPINS = 3
AGENTS = 2
SCENARIO = [
    torch.Tensor([[1, -1], [-1, -1], [1, -1]]),
    torch.Tensor([[-1, 1], [-1, -1], [1, 1]]),
    torch.Tensor([[-1, 1], [-1, -1], [1, -1]]),
    torch.Tensor([[-1, 1], [-1, -1], [1, -1]]),
    torch.Tensor([[-1, 1], [-1, -1], [1, -1]]),
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
        window.previously_bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    window.update(SCENARIO[0])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(), torch.Tensor([]).reshape((3, 0)))
    assert torch.equal(window.current_spins, SCENARIO[0])
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.Tensor([0, 0]))
    assert torch.equal(
        window.newly_bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.Tensor([False, False]).to(dtype=torch.bool))

    # Second update
    assert torch.equal(
        window.previously_bifurcated, torch.Tensor([False, False]).to(dtype=torch.float)
    )
    window.update(SCENARIO[1])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(), torch.Tensor([]).reshape((3, 0)))
    assert torch.equal(window.current_spins, SCENARIO[1])
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.Tensor([0, 0]))
    assert torch.equal(
        window.newly_bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.Tensor([False, False]).to(dtype=torch.bool))

    # Third update
    assert torch.equal(
        window.previously_bifurcated, torch.Tensor([False, False]).to(dtype=torch.float)
    )
    window.update(SCENARIO[2])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(), torch.Tensor([]).reshape((3, 0)))
    assert torch.equal(window.current_spins, SCENARIO[2])
    assert torch.equal(window.final_spins, torch.zeros((3, 2)))
    assert torch.equal(window.stability, torch.Tensor([1, 0]))
    assert torch.equal(
        window.newly_bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.Tensor([False, False]).to(dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.Tensor([True, False]).to(dtype=torch.bool))

    # Fourth update
    assert torch.equal(
        window.previously_bifurcated, torch.Tensor([False, False]).to(dtype=torch.float)
    )
    window.update(SCENARIO[3])
    assert window.must_continue()
    assert window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(), torch.Tensor([[-1], [-1], [1]]))
    assert torch.equal(window.current_spins, SCENARIO[3])
    assert torch.equal(window.final_spins, torch.Tensor([[-1, 0], [-1, 0], [1, 0]]))
    assert torch.equal(window.stability, torch.Tensor([2, 1]))
    assert torch.equal(
        window.newly_bifurcated, torch.Tensor([True, False]).to(dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.Tensor([True, False]).to(dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.Tensor([True, True]).to(dtype=torch.bool))

    # Fourth update
    assert torch.equal(
        window.previously_bifurcated, torch.Tensor([True, False]).to(dtype=torch.float)
    )
    window.update(SCENARIO[4])
    assert not window.must_continue()
    assert window.has_bifurcated_spins()
    assert torch.equal(window.get_bifurcated_spins(), SCENARIO[4])
    assert torch.equal(window.current_spins, SCENARIO[4])
    assert torch.equal(window.final_spins, SCENARIO[4])
    assert torch.equal(window.stability, torch.Tensor([2, 2]))
    assert torch.equal(
        window.newly_bifurcated, torch.Tensor([False, True]).to(dtype=torch.bool)
    )
    assert torch.equal(
        window.bifurcated, torch.Tensor([True, True]).to(dtype=torch.bool)
    )
    assert torch.equal(window.equal, torch.Tensor([True, True]).to(dtype=torch.bool))
