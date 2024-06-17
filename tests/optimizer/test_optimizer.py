import pytest
import torch

from src.simulated_bifurcation import ConvergenceWarning, reset_env, set_env
from src.simulated_bifurcation.core import Ising
from src.simulated_bifurcation.optimizer import (
    SimulatedBifurcationEngine,
    SimulatedBifurcationOptimizer,
)


def test_optimizer():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    ising.minimize(
        20,
        10000,
        False,
        False,
        False,
        use_window=False,
        sampling_period=50,
        convergence_threshold=50,
    )
    assert torch.equal(torch.ones((3, 20)), ising.computed_spins)


def test_optimizer_without_bifurcation():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    with pytest.warns(ConvergenceWarning):
        ising.minimize(
            5,
            10,
            False,
            False,
            False,
            use_window=True,
            sampling_period=50,
            convergence_threshold=50,
        )
    assert torch.equal(
        torch.tensor(
            [
                [1.0, 1.0, -1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        ising.computed_spins,
    )


def test_optimizer_with_window():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    ising.minimize(
        20,
        30000,
        False,
        False,
        False,
        use_window=True,
        sampling_period=20,
        convergence_threshold=20,
    )
    assert torch.equal(torch.ones((3, 20)), ising.computed_spins)


def test_optimizer_with_heating():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    ising.minimize(
        20,
        10000,
        False,
        True,
        False,
        use_window=False,
        sampling_period=50,
        convergence_threshold=50,
    )
    assert torch.equal(torch.ones((3, 20)), ising.computed_spins)


def test_set_optimization_environment():
    torch.manual_seed(42)
    set_env(time_step=0.05, pressure_slope=0.005, heat_coefficient=0.1)
    optimizer = SimulatedBifurcationOptimizer(
        128, 10000, None, SimulatedBifurcationEngine.HbSB, True, 50, 50
    )
    assert optimizer.heat_coefficient == 0.1
    assert optimizer.pressure_slope == 0.005
    assert optimizer.time_step == 0.05
    reset_env()


def test_set_only_one_optimization_variable():
    torch.manual_seed(42)
    set_env(time_step=0.05)
    optimizer = SimulatedBifurcationOptimizer(
        128, 10000, None, SimulatedBifurcationEngine.HbSB, True, 50, 50
    )
    assert optimizer.heat_coefficient == 0.06
    assert optimizer.pressure_slope == 0.01
    assert optimizer.time_step == 0.05
    reset_env()


def test_wrong_value_throws_exception_and_variables_not_updated():
    torch.manual_seed(42)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        set_env(heat_coefficient="Hello world!")
    optimizer = SimulatedBifurcationOptimizer(
        128, 10000, None, SimulatedBifurcationEngine.HbSB, True, 50, 50
    )
    assert optimizer.heat_coefficient == 0.06
    assert optimizer.pressure_slope == 0.01
    assert optimizer.time_step == 0.1


def test_timeout():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    optimizer = SimulatedBifurcationOptimizer(
        128, None, 3.0, SimulatedBifurcationEngine.HbSB, True, 50, 50
    )
    optimizer.run_integrator(ising.as_simulated_bifurcation_tensor(), False)
    assert optimizer.simulation_time > 3.0


def test_window():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    optimizer = SimulatedBifurcationOptimizer(
        1, 100000, None, SimulatedBifurcationEngine.HbSB, True, 1, 1
    )
    optimizer.run_integrator(ising.as_simulated_bifurcation_tensor(), True)


def test_max_steps():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    optimizer = SimulatedBifurcationOptimizer(
        1, 10, None, SimulatedBifurcationEngine.HbSB, True, 50, 50
    )
    optimizer.run_integrator(ising.as_simulated_bifurcation_tensor(), False)
    assert optimizer.step == 10


def test_no_stop_criterion():
    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    optimizer = SimulatedBifurcationOptimizer(
        1, None, None, SimulatedBifurcationEngine.HbSB, True, 50, 50
    )
    with pytest.raises(ValueError, match="No stopping criterion provided."):
        optimizer.run_integrator(ising.as_simulated_bifurcation_tensor(), False)


def test_keyboard_interrupt():
    class SimulatedBifurcationOptimizerTest(SimulatedBifurcationOptimizer):
        def _step_update(self) -> None:
            super()._step_update()
            if self.step >= 1000:
                raise KeyboardInterrupt

    torch.manual_seed(42)
    J = torch.tensor(
        [
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1],
        ],
        dtype=torch.float32,
    )
    h = torch.tensor([1, 0, -2], dtype=torch.float32)
    ising = Ising(J, h)
    optimizer = SimulatedBifurcationOptimizerTest(
        20, 10000, None, SimulatedBifurcationEngine.bSB, True, 50, 50
    )
    with pytest.warns(
        RuntimeWarning,
        match="Simulation interrupted by user. Current spins will be returned.",
    ):
        spins = optimizer.run_integrator(ising.as_simulated_bifurcation_tensor(), False)
        assert isinstance(spins, torch.Tensor)
        assert (4, 20) == tuple(spins.shape)
        assert torch.all(torch.abs(spins) == 1.0)
        assert optimizer.step < 10000
