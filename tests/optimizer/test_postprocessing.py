import torch

from src.simulated_bifurcation.optimizer.postprocessing import Postprocessing


def test_reconstruct_spins():
    optimized_spins = torch.tensor([[1, 1, -1, 1], [-1, 1, -1, -1], [1, 1, -1, 1]])
    pre_solved_spins = torch.tensor([1, 0, 0, -1, 0])
    assert torch.equal(
        torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, -1, 1],
                [-1, 1, -1, -1],
                [-1, -1, -1, -1],
                [1, 1, -1, 1],
            ]
        ),
        Postprocessing.reconstruct_spins(optimized_spins, pre_solved_spins),
    )
