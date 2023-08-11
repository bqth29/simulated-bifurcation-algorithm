import logging
from typing import Tuple

import torch
from numpy import minimum
from tqdm import tqdm

from .optimization_variables import OptimizationVariable
from .optimizer_mode import OptimizerMode
from .stop_window import StopWindow
from .symplectic_integrator import SymplecticIntegrator

LOGGER = logging.getLogger("simulated_bifurcation_optimizer")
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.set_name(logging.WARN)
CONSOLE_HANDLER.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
LOGGER.addHandler(CONSOLE_HANDLER)


class SimulatedBifurcationOptimizer:

    """
    The Simulated Bifurcation (SB) algorithm relies on
    Hamiltonian/quantum mechanics to find local minima of
    Ising problems. The spins dynamics is simulated using
    a first order symplectic integrator.

    There are different version of the SB algorithm:
    - the ballistic Simulated Bifurcation (bSB) which uses the particles'
    position for the matrix computations (usually faster but less accurate)
    - the discrete Simulated Bifurcation (dSB) which uses the particles'
    spin for the matrix computations (usually slower but more accurate)
    - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
    algorithm with a supplementary non-symplectic term to refine the model
    - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
    algorithm with a supplementary non-symplectic term to refine the model

    To stop the iterations of the symplectic integrator, a number of maximum
    steps needs to be specified. However, a refined way to stop is also possible
    using a window that checks that the spins have not changed among a set
    number of previous steps. In practice, a every fixed number of steps
    (called a sampling period) the current spins will be compared to the
    previous ones. If they remain constant throughout a certain number of
    consecutive samplings (called the convergence threshold), the spins are
    considered to have bifurcated and the algorithm stops.

    Finally, it is possible to make several particle vectors at the same
    time (each one being called an agent). As the vectors are randomly
    initialized, using several agents helps to explore the solution space
    and increases the probability of finding a better solution, though it
    also slightly increases the computation time. In the end, only the best
    spin vector (energy-wise) is kept and used as the new Ising model's
    ground state.
    """

    def __init__(
        self,
        agents: int,
        max_steps: int,
        mode: OptimizerMode,
        heated: bool,
        verbose: bool,
        sampling_period: int,
        convergence_threshold: int,
    ) -> None:
        # Optimizer setting
        self.mode = mode
        self.window = None
        self.symplectic_integrator = None
        self.heat_coefficient = OptimizationVariable.HEAT_COEFFICIENT.get()
        self.heated = heated
        self.verbose = verbose
        # Simulation parameters
        self.time_step = OptimizationVariable.TIME_STEP.get()
        self.agents = agents
        self.pressure_slope = OptimizationVariable.PRESSURE_SLOPE.get()
        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps

    def __reset(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.__init_progress_bar(self.max_steps, self.verbose)
        self.__init_symplectic_integrator(matrix)
        self.__init_window(matrix, use_window)
        self.__init_quadratic_scale_parameter(matrix)
        self.run = True
        self.step = 0

    def __init_progress_bar(self, max_steps: int, verbose: bool) -> None:
        self.iterations_progress = tqdm(
            total=max_steps,
            desc="Iterations",
            disable=not verbose,
            smoothing=0.1,
            mininterval=0.5,
        )

    def __init_quadratic_scale_parameter(self, matrix: torch.Tensor):
        self.quadratic_scale_parameter = (
            0.5 * (matrix.shape[0] - 1) ** 0.5 / (torch.sqrt(torch.sum(matrix**2)))
        )

    def __init_window(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.window = StopWindow(
            matrix.shape[0],
            self.agents,
            self.convergence_threshold,
            matrix.dtype,
            str(matrix.device),
            (self.verbose and use_window),
        )

    def __init_symplectic_integrator(self, matrix: torch.Tensor) -> None:
        self.symplectic_integrator = SymplecticIntegrator(
            (matrix.shape[0], self.agents), self.mode, matrix.dtype, str(matrix.device)
        )

    def __step_update(self) -> None:
        self.step += 1
        self.iterations_progress.update()

    def __check_stop(self, use_window: bool) -> None:
        if use_window and self.__do_sampling:
            self.run = self.window.must_continue()
        if self.step >= self.max_steps:
            self.run = False

    @property
    def __do_sampling(self) -> bool:
        return self.step % self.sampling_period == 0

    def __close_progress_bars(self):
        self.iterations_progress.close()
        self.window.progress.close()

    def __symplectic_update(
        self, matrix: torch.Tensor, use_window: bool
    ) -> torch.Tensor:
        while self.run:
            if self.heated:
                momentum_copy = self.symplectic_integrator.momentum.clone()

            (
                momentum_coefficient,
                position_coefficient,
                quadratic_coefficient,
            ) = self.__compute_symplectic_coefficients()
            self.symplectic_integrator.step(
                momentum_coefficient,
                position_coefficient,
                quadratic_coefficient,
                matrix,
            )

            if self.heated:
                self.__heat(momentum_copy)

            self.__step_update()
            sampled_spins = self.symplectic_integrator.sample_spins()
            if use_window and self.__do_sampling:
                self.window.update(sampled_spins)

            self.__check_stop(use_window)

        return sampled_spins

    def __heat(self, momentum_copy: torch.Tensor) -> None:
        torch.add(
            self.symplectic_integrator.momentum,
            momentum_copy,
            alpha=self.time_step * self.heat_coefficient,
            out=self.symplectic_integrator.momentum,
        )

    def __compute_symplectic_coefficients(self) -> Tuple[float, float, float]:
        pressure = self.__pressure
        position_coefficient = self.time_step
        momentum_coefficient = self.time_step * (pressure - 1.0)
        quadratic_coefficient = self.time_step * self.quadratic_scale_parameter
        return momentum_coefficient, position_coefficient, quadratic_coefficient

    @property
    def __pressure(self):
        return minimum(self.time_step * self.step * self.pressure_slope, 1.0)

    def run_integrator(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:
        """
        Runs the Simulated Bifurcation (SB) algorithm.
        """
        self.__reset(matrix, use_window)
        spins = self.__symplectic_update(matrix, use_window)
        self.__close_progress_bars()
        return self.get_final_spins(spins, use_window)

    def get_final_spins(self, spins: torch.Tensor, use_window: bool) -> torch.Tensor:
        """
        Returns the final spins retrieved at the end of the
        Simulated Bifurcation (SB) algorithm.

        If the stop window was used, it returns the bifurcated agents if any,
        otherwise the actual final spins are returned.

        If the stop window was not used, the final spins are returned.
        """
        if use_window:
            if not self.window.has_bifurcated_spins():
                LOGGER.warning(
                    "No agent has converged. Returned final positions' signs instead."
                )
            return self.window.get_bifurcated_spins(spins)
        else:
            return spins
