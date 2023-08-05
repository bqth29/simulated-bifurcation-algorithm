import pytest
import torch

from src.simulated_bifurcation import reset_env, set_env
from src.simulated_bifurcation.ising_core import IsingCore
from src.simulated_bifurcation.optimizer import (
    OptimizerMode,
    SimulatedBifurcationOptimizer,
)

torch.manual_seed(42)


def test_optimizer():
    J = torch.Tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ]
    )
    h = torch.Tensor([1, 0, -2])
    ising = IsingCore(J, h)
    ising.optimize(50, 50, 10000, 20, False, False, False, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5


def test_optimizer_without_bifurcation():
    J = torch.Tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ]
    )
    h = torch.Tensor([1, 0, -2])
    ising = IsingCore(J, h)
    ising.optimize(50, 50, 10, 20, True, False, False, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5


def test_optimizer_with_window():
    J = torch.Tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ]
    )
    h = torch.Tensor([1, 0, -2])
    ising = IsingCore(J, h)
    ising.optimize(20, 20, 30000, 20, True, False, False, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5


def test_optimizer_with_heating():
    J = torch.Tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ]
    )
    h = torch.Tensor([1, 0, -2])
    ising = IsingCore(J, h)
    ising.optimize(50, 50, 10000, 20, False, False, True, False)
    assert torch.equal(torch.Tensor([1, 1, 1]), ising.ground_state)
    assert ising.energy == -11.5


def test_set_optimization_environment():
    set_env(time_step=0.05, pressure_slope=0.005, heat_coefficient=0.1)
    optimizer = SimulatedBifurcationOptimizer(
        50, 50, 10000, 128, OptimizerMode.BALLISTIC, True, True
    )
    assert optimizer.heat_coefficient == 0.1
    assert optimizer.pressure_slope == 0.005
    assert optimizer.time_step == 0.05
    reset_env()


def test_set_only_one_optimization_variable():
    set_env(time_step=0.05)
    optimizer = SimulatedBifurcationOptimizer(
        50, 50, 10000, 128, OptimizerMode.BALLISTIC, True, True
    )
    assert optimizer.heat_coefficient == 0.06
    assert optimizer.pressure_slope == 0.01
    assert optimizer.time_step == 0.05
    reset_env()


def test_wrong_value_throws_exception_and_variables_not_updated():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        set_env(heat_coefficient="Hello world!")
    optimizer = SimulatedBifurcationOptimizer(
        50, 50, 10000, 128, OptimizerMode.BALLISTIC, True, True
    )
    assert optimizer.heat_coefficient == 0.06
    assert optimizer.pressure_slope == 0.01
    assert optimizer.time_step == 0.1
