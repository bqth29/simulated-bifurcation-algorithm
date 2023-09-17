import torch

from src.simulated_bifurcation.models import QUBO


def test_qubo():
    torch.manual_seed(42)
    Q = torch.tensor([[1, -2, 3], [0, -4, 1], [0, 0, 2]])
    model = QUBO(Q)
    binary_vector, value = model.minimize(agents=10, verbose=False, best_only=True)
    assert torch.equal(torch.tensor([1.0, 1.0, 0.0]), binary_vector)
    assert -5.0 == value
    binary_vector, value = model.maximize(agents=10, verbose=False, best_only=True)
    assert torch.equal(torch.tensor([1.0, 0.0, 1.0]), binary_vector)
    assert 6.0 == value
