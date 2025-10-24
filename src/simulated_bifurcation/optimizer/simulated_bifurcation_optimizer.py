import warnings
from time import time
from typing import Optional, Union

import torch
from numpy import minimum
from tqdm.auto import tqdm

from ..core.tensor_bearer import TensorBearer
from .environment import ENVIRONMENT
from .simulated_bifurcation_engine import SimulatedBifurcationEngine
from .stop_window import StopWindow
from .symplectic_integrator import SymplecticIntegrator


class ConvergenceWarning(Warning):
    def __str__(self) -> str:
        return "No agent has converged. Returned signs of final positions instead."


class SimulatedBifurcationOptimizer(TensorBearer):
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
        heated: bool,
        verbose: bool,
        sampling_period: int,
        convergence_threshold: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        # Optimizer setting
        self.engine = engine
        self.window = None
        self.symplectic_integrator = None
        self.heat_coefficient = ENVIRONMENT.heat_coefficient
        self.heated = heated
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

    def __reset(self, matrix: torch.Tensor, early_stopping: bool) -> None:
        self.__init_progress_bars()
        self.__init_symplectic_integrator(matrix)
        self.__init_window(matrix, early_stopping)
        self.__init_quadratic_scale_parameter(matrix)
        self.run = True
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

    def __init_window(self, matrix: torch.Tensor, early_stopping: bool) -> None:
        self.window = StopWindow(
            matrix,
            self.agents,
            self.convergence_threshold,
            matrix.dtype,
            matrix.device,
            (self.verbose and early_stopping),
        )

    def __init_symplectic_integrator(self, matrix: torch.Tensor) -> None:
        self.symplectic_integrator = SymplecticIntegrator(
            self.agents,
            self.time_step,
            self.pressure_slope,
            self.heat_coefficient,
            self.engine.activation_function,
            self.heated,
            matrix,
            self.dtype,
            self.device,
        )

    def _check_stop(self, early_stopping: bool) -> None:
        if early_stopping and self.__must_sample_spins():
            self.run = self.window.must_continue()
            if not self.run:
                return
        if self.symplectic_integrator.step >= self.max_steps:
            self.run = False
            return
        previous_time = self.simulation_time
        self.simulation_time = time() - self.start_time
        time_update = min(
            self.simulation_time - previous_time, self.timeout - previous_time
        )
        self.time_progress.update(time_update)
        if self.simulation_time > self.timeout:
            self.run = False
            return

    def __must_sample_spins(self) -> bool:
        return self.symplectic_integrator.step % self.sampling_period == 0

    def __close_progress_bars(self):
        self.iterations_progress.close()
        self.time_progress.close()
        self.window.progress.close()

    def __symplectic_update(self, early_stopping: bool) -> torch.Tensor:
        self.start_time = time()
        try:
            while self.run:
                self.symplectic_integrator.integration_step()
                self.iterations_progress.update()
                if early_stopping and self.__must_sample_spins():
                    sampled_spins = self.symplectic_integrator.sample_spins()
                    self.window.update(sampled_spins)

                self._check_stop(early_stopping)
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

    def run_integrator(
        self, matrix: torch.Tensor, early_stopping: bool
    ) -> torch.Tensor:
        """
        Runs the Simulated Bifurcation (SB) algorithm. Given an input matrix,
        the SB algorithm aims at finding the ground state of the Ising model
        defined from this matrix, i.e. the {-1, +1}-vector that minimizes the
        Ising energy defined as `-0.5 * ΣΣ J(i,j)x(i)x(j)`, where `J`
        designates the matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            The matrix that defines the Ising model to optimize.
        early_stopping : bool
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
            and not early_stopping
        ):
            raise ValueError("No stopping criterion provided.")
        self.__reset(matrix, early_stopping)
        spins = self.__symplectic_update(early_stopping)
        self.__close_progress_bars()
        return self.get_final_spins(spins, early_stopping)

    def get_final_spins(
        self, spins: torch.Tensor, early_stopping: bool
    ) -> torch.Tensor:
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
        early_stopping : bool
            Whether the stop window was used or not.

        Returns
        -------
        torch.Tensor
        """
        if early_stopping:
            if not self.window.has_bifurcated_spins():
                warnings.warn(ConvergenceWarning(), stacklevel=2)
            return self.window.get_bifurcated_spins(spins)
        else:
            return spins
