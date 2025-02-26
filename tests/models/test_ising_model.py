import torch

from src.simulated_bifurcation.models import Ising


def test_ising():
    torch.manual_seed(42)
    J = torch.tensor([[0, -2, 3], [-2, 0, 1], [3, 1, 0]])
    h = torch.tensor([1, -4, 2])
    model = Ising(J, h, dtype=torch.float32, device=torch.device("cpu"))
    spin_vector, value = model.optimize(
        agents=10,
        verbose=False,
        best_only=True,
        mode="ballistic",
    )
    assert torch.equal(
        torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32), spin_vector
    )
    assert -11.0 == value
