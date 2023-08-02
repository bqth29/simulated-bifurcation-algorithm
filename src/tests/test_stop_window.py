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
    assert torch.all(window.current_spins == torch.zeros((3, 2)))
    assert torch.all(window.final_spins == torch.zeros((3, 2)))


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
    assert torch.all(window.previously_bifurcated == torch.Tensor([False, False]))
    window.update(SCENARIO[0])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.all(window.get_bifurcated_spins() == torch.Tensor([]))
    assert torch.all(window.current_spins == SCENARIO[0])
    assert torch.all(window.final_spins == torch.zeros((3, 2)))
    assert torch.all(window.stability == torch.Tensor([0, 0]))
    assert torch.all(window.newly_bifurcated == torch.Tensor([False, False]))
    assert torch.all(window.bifurcated == torch.Tensor([False, False]))
    assert torch.all(window.equal == torch.Tensor([False, False]))

    # Second update
    assert torch.all(window.previously_bifurcated == torch.Tensor([False, False]))
    window.update(SCENARIO[1])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.all(window.get_bifurcated_spins() == torch.Tensor([]))
    assert torch.all(window.current_spins == SCENARIO[1])
    assert torch.all(window.final_spins == torch.zeros((3, 2)))
    assert torch.all(window.stability == torch.Tensor([0, 0]))
    assert torch.all(window.newly_bifurcated == torch.Tensor([False, False]))
    assert torch.all(window.bifurcated == torch.Tensor([False, False]))
    assert torch.all(window.equal == torch.Tensor([False, False]))

    # Third update
    assert torch.all(window.previously_bifurcated == torch.Tensor([False, False]))
    window.update(SCENARIO[2])
    assert window.must_continue()
    assert not window.has_bifurcated_spins()
    assert torch.all(window.get_bifurcated_spins() == torch.Tensor([]))
    assert torch.all(window.current_spins == SCENARIO[2])
    assert torch.all(window.final_spins == torch.zeros((3, 2)))
    assert torch.all(window.stability == torch.Tensor([1, 0]))
    assert torch.all(window.newly_bifurcated == torch.Tensor([False, False]))
    assert torch.all(window.bifurcated == torch.Tensor([False, False]))
    assert torch.all(window.equal == torch.Tensor([True, False]))

    # Fourth update
    assert torch.all(window.previously_bifurcated == torch.Tensor([False, False]))
    window.update(SCENARIO[3])
    assert window.must_continue()
    assert window.has_bifurcated_spins()
    assert torch.all(window.get_bifurcated_spins() == torch.Tensor([[-1], [-1], [1]]))
    assert torch.all(window.current_spins == SCENARIO[3])
    assert torch.all(window.final_spins == torch.Tensor([[-1, 0], [-1, 0], [1, 0]]))
    assert torch.all(window.stability == torch.Tensor([2, 1]))
    assert torch.all(window.newly_bifurcated == torch.Tensor([True, False]))
    assert torch.all(window.bifurcated == torch.Tensor([True, False]))
    assert torch.all(window.equal == torch.Tensor([True, True]))

    # Fourth update
    assert torch.all(window.previously_bifurcated == torch.Tensor([True, False]))
    window.update(SCENARIO[4])
    assert not window.must_continue()
    assert window.has_bifurcated_spins()
    assert torch.all(window.get_bifurcated_spins() == SCENARIO[4])
    assert torch.all(window.current_spins == SCENARIO[4])
    assert torch.all(window.final_spins == SCENARIO[4])
    assert torch.all(window.stability == torch.Tensor([2, 2]))
    assert torch.all(window.newly_bifurcated == torch.Tensor([False, True]))
    assert torch.all(window.bifurcated == torch.Tensor([True, True]))
    assert torch.all(window.equal == torch.Tensor([True, True]))
