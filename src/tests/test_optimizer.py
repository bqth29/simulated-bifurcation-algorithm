import pytest
import torch
from src.simulated_bifurcation import Ising, SimulatedBifurcationOptimizer, set_env, reset_env


torch.manual_seed(42)

def test_optimizer():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(50, 50, 10000, 20, False, False, False, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5

def test_optimizer_without_bifurcation():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(50, 50, 10, 20, True, False, False, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5

def test_optimizer_with_window():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(20, 20, 30000, 20, True, False, False, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5

def test_optimizer_with_heating():
    J = torch.Tensor([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    h = torch.Tensor([1, 0, -2])
    ising = Ising(J, h)
    ising.optimize(50, 50, 10000, 20, False, False, True, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5

def test_set_optimization_environment():
    set_env(time_step=.05, pressure_slope=.005, heat_coefficient=.1)
    optimizer = SimulatedBifurcationOptimizer(50, 50, 10000, 128, True, True, True)
    assert optimizer.heat_coefficient == .1
    assert optimizer.pressure_slope == .005
    assert optimizer.time_step == .05
    reset_env()

def test_set_only_one_optimization_variable():
    set_env(time_step=.05)
    optimizer = SimulatedBifurcationOptimizer(50, 50, 10000, 128, True, True, True)
    assert optimizer.heat_coefficient == .06
    assert optimizer.pressure_slope == .01
    assert optimizer.time_step == .05
    reset_env()

def test_wrong_value_throws_exception_and_variables_not_updated():
    with pytest.raises(TypeError):
        set_env(heat_coefficient='Hello world!')
    optimizer = SimulatedBifurcationOptimizer(50, 50, 10000, 128, True, True, True)
    assert optimizer.heat_coefficient == .06
    assert optimizer.pressure_slope == .01
    assert optimizer.time_step == .1
