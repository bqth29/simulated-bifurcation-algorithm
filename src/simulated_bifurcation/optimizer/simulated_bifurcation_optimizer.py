import logging
import warnings
from time import time
from typing import Optional, Tuple

import torch
from numpy import minimum
from tqdm.auto import tqdm

from .environment import ENVIRONMENT
from .simulated_bifurcation_engine import SimulatedBifurcationEngine
from .stop_window import StopWindow
from .symplectic_integrator import SymplecticIntegrator

LOGGER = logging.getLogger("simulated_bifurcation_optimizer")
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.set_name(logging.WARN)
CONSOLE_HANDLER.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
LOGGER.addHandler(CONSOLE_HANDLER)


class ConvergenceWarning(Warning):
    def __str__(self) -> str:
        return "No agent has converged. Returned signs of final positions instead."


class SimulatedBifurcationOptimizer:
    """
    The Simulated Bifurcation (SB) algorithm relies on
    Hamiltonian/quantum mechanics to find local minima of
    Ising problems. The spins dynamics is simulated using
    a first order symplectic integrator.

    There are 4 different version of the SB algorithm:

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
        max_steps: Optional[int],
        timeout: Optional[float],
        engine: SimulatedBifurcationEngine,
        verbose: bool,
        sampling_period: int,
        convergence_threshold: int,
    ) -> None:
        # Optimizer setting
        self.engine = engine
        self.window = None
        self.symplectic_integrator = None
        self.heat_coefficient = ENVIRONMENT.heat_coefficient
        self.heated = engine.heated
        self.verbose = verbose
        self.start_time = None
        self.simulation_time = None
        # Simulation parameters
        self.time_step = ENVIRONMENT.time_step
        self.agents = agents
        self.pressure_slope = ENVIRONMENT.pressure_slope
        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps if max_steps is not None else float("inf")
        self.timeout = timeout if timeout is not None else float("inf")

    def __reset(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.__init_progress_bars()
        self.__init_symplectic_integrator(matrix)
        self.__init_window(matrix, use_window)
        self.__init_quadratic_scale_parameter(matrix)
        self.run = True
        self.step = 0
        self.start_time = None
        self.simulation_time = 0

    def __init_progress_bars(self) -> None:
        self.iterations_progress = tqdm(
            total=self.max_steps,
            desc="🔁 Iterations       ",
            disable=not self.verbose or self.max_steps == float("inf"),
            smoothing=0.1,
            mininterval=0.5,
            unit=" steps",
        )
        self.time_progress = tqdm(
            total=self.timeout,
            desc="⏳ Simulation time  ",
            disable=not self.verbose or self.timeout == float("inf"),
            smoothing=0.1,
            mininterval=0.5,
            bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} seconds",
        )

    def __init_quadratic_scale_parameter(self, matrix: torch.Tensor):
        self.quadratic_scale_parameter = (
            0.5 * (matrix.shape[0] - 1) ** 0.5 / (torch.sqrt(torch.sum(matrix**2)))
        )

    def __init_window(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.window = StopWindow(
            matrix,
            self.agents,
            self.convergence_threshold,
            matrix.dtype,
            matrix.device,
            (self.verbose and use_window),
        )

    def __init_symplectic_integrator(self, matrix: torch.Tensor) -> None:
        self.symplectic_integrator = SymplecticIntegrator(
            (matrix.shape[0], self.agents),
            self.engine.activation_function,
            matrix.dtype,
            matrix.device,
        )

    def _step_update(self) -> None:
        self.step += 1
        self.iterations_progress.update()

    def __check_stop(self, use_window: bool) -> None:
        if use_window and self.__do_sampling:
            self.run = self.window.must_continue()
            if not self.run:
                LOGGER.info("Optimizer stopped. Reason: all agents converged.")
                return
        if self.step >= self.max_steps:
            self.run = False
            LOGGER.info(
                "Optimizer stopped. Reason: maximum number of iterations reached."
            )
            return
        previous_time = self.simulation_time
        self.simulation_time = time() - self.start_time
        time_update = min(
            self.simulation_time - previous_time, self.timeout - previous_time
        )
        self.time_progress.update(time_update)
        if self.simulation_time > self.timeout:
            self.run = False
            LOGGER.info("Optimizer stopped. Reason: computation timeout reached.")
            return

    @property
    def __do_sampling(self) -> bool:
        return self.step % self.sampling_period == 0

    def __close_progress_bars(self):
        self.iterations_progress.close()
        self.time_progress.close()
        self.window.progress.close()

    def __symplectic_update(
        self,
        matrix: torch.Tensor,
        use_window: bool,
    ) -> torch.Tensor:
        self.start_time = time()
        try:
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

                self._step_update()
                if use_window and self.__do_sampling:
                    sampled_spins = self.symplectic_integrator.sample_spins()
                    self.window.update(sampled_spins)

                self.__check_stop(use_window)
        except KeyboardInterrupt:
            warnings.warn(
                RuntimeWarning(
                    "Simulation interrupted by user. Current spins will be returned."
                ),
                stacklevel=2,
            )
        finally:
            sampled_spins = self.symplectic_integrator.sample_spins()
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
        Runs the Simulated Bifurcation (SB) algorithm. Given an input matrix,
        the SB algorithm aims at finding the groud state of the Ising model
        defined from this matrix, i.e. the {-1, +1}-vector that minimizes the
        Ising energy defined as `-0.5 * ΣΣ J(i,j)x(i)x(j)`, where `J`
        designates the matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            The matrix that defines the Ising model to optimize.
        use_window : bool
            Whether to use a stop window or not to perform early-stopping.

        Returns
        -------
        torch.Tensor
            The optimized spins. The shape is (dimension of the matrix, agents).

        Raises
        ------
        ValueError
            If no stopping criterion was provided, the algorithm will not start.
        """
        if (
            self.max_steps == float("inf")
            and self.timeout == float("inf")
            and not use_window
        ):
            raise ValueError("No stopping criterion provided.")
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

        Parameters
        ----------
        spins : torch.Tensor
            The spins returned by the Simulated Bifurcation algorithm.
        use_window : bool
            Whether the stop window was used or not.

        Returns
        -------
        torch.Tensor
        """
        if use_window:
            if not self.window.has_bifurcated_spins():
                warnings.warn(ConvergenceWarning(), stacklevel=2)
            return self.window.get_bifurcated_spins(spins)
        else:
            return spins
