import torch
from src.simulated_bifurcation import Ising


torch.manual_seed(42)

def test_optimizer():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(50, 50, 10000, 20, False, False, False, False)
    assert torch.all(ising.ground_state == torch.Tensor([1, 1, 1]))
    assert ising.energy == -11.5

def test_optimizer_without_bifurcation():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(50, 50, 10, 20, True, False, False, False)
    assert torch.all(ising.ground_state == torch.Tensor([1, 1, 1]))
    assert ising.energy == -11.5

def test_optimizer_with_window():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(20, 20, 30000, 20, True, False, False, False)
    assert torch.all(ising.ground_state == torch.Tensor([1, 1, 1]))
    assert ising.energy == -11.5

def test_optimizer_with_heating():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(50, 50, 10000, 20, False, False, True, False)
    assert torch.all(ising.ground_state == torch.Tensor([1, 1, 1]))
    assert ising.energy == -11.5