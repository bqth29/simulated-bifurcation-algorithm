from typing import Tuple, Union
import torch
from tqdm import tqdm
from numpy import minimum
from enum import Enum


class OptimizerMode(Enum):
    BALLISTIC = ('ballistic', torch.nn.Identity())
    DISCRETE = ('discrete', torch.sign)

    def __str__(self) -> str:
        return self.mode
    
    def __repr__(self) -> str:
        return self.mode

    def __call__(self, input: torch.Tensor):
        return self.activation_function(input)
    
    @property
    def mode(self) -> str: return self.value[0]

    @property
    def activation_function(self) -> str: return self.value[1]


class SymplecticIntegrator:

    def __init__(self, shape: Tuple[int, int], mode: OptimizerMode, dtype: torch.dtype, device: str):
        self.momentum = SymplecticIntegrator.__init_oscillator(shape, dtype, device)
        self.position = SymplecticIntegrator.__init_oscillator(shape, dtype, device)
        self.activation_function = mode.activation_function

    @classmethod
    def __init_oscillator(cls, shape: Tuple[int, int], dtype: torch.dtype, device: str):
        return 2 * torch.rand(size=shape, device=device, dtype=dtype) - 1

    def momentum_update(self, coefficient: float) -> None:
        torch.add(self.momentum, coefficient * self.position, out=self.momentum)
    
    def position_update(self, coefficient: float) -> None:
        torch.add(self.position, coefficient * self.momentum, out=self.position)

    def quadratic_position_update(self, coefficient: float, matrix: torch.Tensor) -> None:
        torch.add(self.position, coefficient * matrix @ self.activation_function(self.momentum), out=self.position)

    def step(self, momentum_coefficient: float, position_coefficient: float, quadratic_coefficient: float, matrix: torch.Tensor) -> None:
        self.momentum_update(momentum_coefficient)
        self.position_update(position_coefficient)
        self.quadratic_position_update(quadratic_coefficient, matrix)

    def sample_spins(self) -> torch.Tensor:
        return torch.sign(self.momentum)


class StopWindow:

    def __init__(self, n_spins: int, n_agents: int, convergence_threshold: int, sampling_period: int, dtype: torch.dtype, device: str, verbose: bool) -> None:
        self.n_spins = n_spins
        self.n_agents = n_agents
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.dtype = dtype
        self.device = device
        self.__init_tensors()
        self.current_spins = self.__init_spins()
        self.final_spins = self.__init_spins()
        self.progress = self.__init_progress_bar(verbose)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_spins, self.n_agents)
    
    def __init_progress_bar(self, verbose: bool) -> tqdm:
        return tqdm(
            total=self.n_agents,
            desc='Bifurcated agents',
            disable=not verbose,
            smoothing=0
        )
    
    def __init_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.n_agents, device=self.device, dtype=dtype)

    def __init_tensors(self) -> None:
        self.stability = self.__init_tensor(self.dtype)
        self.newly_bifurcated = self.__init_tensor(bool)
        self.previously_bifurcated = self.__init_tensor(bool)
        self.bifurcated = self.__init_tensor(bool)
        self.equal = self.__init_tensor(bool)

    def __init_spins(self) -> torch.Tensor:
        return torch.zeros(size=self.shape, dtype=self.dtype, device=self.device)

    def update(self, sampled_spins: torch.Tensor):
        self.compare_spins(sampled_spins)
        self.update_stability_streak()
        self.update_bifurcated_spins()
        self.set_newly_bifurcated_spins()
        self.set_previously_bifurcated_spins()
        self.update_final_spins(sampled_spins)
        self.store_spins(sampled_spins)
        self.progress.update(self.get_number_newly_bifurcated_agents())

    def update_final_spins(self, sampled_spins) -> None:
        self.final_spins[:, self.newly_bifurcated] = torch.sign(
            sampled_spins[:, self.newly_bifurcated]
        )

    def set_previously_bifurcated_spins(self) -> None:
        self.previously_bifurcated = self.bifurcated * 1.

    def set_newly_bifurcated_spins(self) -> None:
        torch.logical_xor(
            self.bifurcated,
            self.previously_bifurcated,
            out=self.newly_bifurcated
        )

    def update_bifurcated_spins(self) -> None:
        torch.eq(
            self.stability,
            self.convergence_threshold - 1,
            out=self.bifurcated
        )

    def update_stability_streak(self) -> None:
        self.stability[torch.logical_and(self.equal, self.not_bifurcated)] += 1
        self.stability[torch.logical_and(self.not_equal, self.not_bifurcated)] = 0

    @property
    def not_equal(self) -> torch.Tensor:
        return torch.logical_not(self.equal)

    @property
    def not_bifurcated(self) -> torch.Tensor:
        return torch.logical_not(self.bifurcated)

    def compare_spins(self, sampled_spins: torch.Tensor) -> None:
        torch.eq(
            torch.einsum('ik, ik -> k', self.current_spins, sampled_spins),
            self.n_spins,
            out=self.equal
        )

    def store_spins(self, sampled_spins: torch.Tensor) -> None:
        torch.sign(sampled_spins, out=self.current_spins)

    def get_number_newly_bifurcated_agents(self) -> int:
        return self.newly_bifurcated.sum().item()
    
    def must_stop(self) -> bool:
        return torch.any(self.stability < self.convergence_threshold - 1)
    
    def has_bifurcated_spins(self) -> bool:
        torch.any(self.bifurcated)

    def get_bifurcated_spins(self) -> torch.Tensor:
        return self.final_spins[:, self.bifurcated]


class Optimizer:

    def __init__(
        self,
        time_step: float,
        convergence_threshold: int,
        sampling_period: int,
        max_steps: int,
        agents: int,
        pressure_slope: float,
        gerschgorin: bool,
        ballistic: bool,
        heat_parameter: float,
        verbose: bool
    ) -> None:

        # Optimizer setting
        self.mode = OptimizerMode.BALLISTIC if ballistic else OptimizerMode.DISCRETE
        self.window = None
        self.symplectic_integrator = None
        self.heat_parameter = heat_parameter
        self.heated = heat_parameter is not None
        self.verbose = verbose

        # Simulation parameters
        self.time_step = time_step
        self.agents = agents
        self.pressure_slope = pressure_slope

        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps

        # Quantum parameters
        self.gerschgorin = gerschgorin

    def __init_progress_bar(self, max_steps: int, verbose: bool) -> tqdm:
        return tqdm(
            total=max_steps, desc='Iterations',
            disable=not verbose, smoothing=0.1,
            mininterval=0.5
        )

    def confine(self) -> None:
        self.symplectic_integrator.position[torch.abs(self.symplectic_integrator.momentum) > 1.] = 0
        torch.clip(self.symplectic_integrator.momentum, -1., 1., out=self.symplectic_integrator.momentum)

    def reset(self, matrix: torch.Tensor) -> None:
        self.iterations_progress = self.__init_progress_bar(self.max_steps, self.verbose)
        self.symplectic_integrator = self.__init_symplectic_integrator(matrix)
        self.window = self.__init_window(matrix)
        self.run = True
        self.step = 0
        self.__init_xi0(matrix)

    def __init_xi0(self, matrix: torch.Tensor):
        if not self.gerschgorin:
            self.xi0 = 0.7 / \
                (torch.std(matrix) * (matrix.shape[0])**(1/2))
        else:
            self.xi0 = 1. / torch.max(
                torch.sum(torch.abs(matrix), axis=1))

    def __init_window(self, matrix: torch.Tensor) -> StopWindow:
        return StopWindow(
            matrix.shape[0], self.agents,
            self.convergence_threshold, self.sampling_period,
            matrix.dtype, str(matrix.device), self.verbose
        )

    def __init_symplectic_integrator(self, matrix: torch.Tensor) -> SymplecticIntegrator:
        return SymplecticIntegrator(
            (matrix.shape[0], self.agents),
            self.mode, matrix.dtype, str(matrix.device)
        )

    def step_update(self) -> None:
        self.step += 1
        self.iterations_progress.update()

    def check_stop(self, use_window: bool) -> None:
        if use_window and self.__go_sampling:
            self.run = self.window.must_stop()
        if self.step >= self.max_steps:
            self.run = False

    @property
    def __go_sampling(self) -> bool:
        return self.step % self.sampling_period == 0

    def run_integrator(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:
        self.reset(matrix)
        spins = self.symplectic_update(matrix, use_window)
        self.close_progress_bars()
        return self.get_final_spins(spins, use_window)

    def close_progress_bars(self):
        self.iterations_progress.close()
        self.window.progress.close()

    def symplectic_update(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:

        while self.run:

            if self.heated: position_stash = self.position.clone().detach()

            momentum_coefficient, position_coefficient, quadratic_coefficient = self.compute_symplectic_coefficients()
            self.symplectic_integrator.step(momentum_coefficient, position_coefficient, quadratic_coefficient, matrix)
            self.confine()

            if self.heated: self.heat(position_stash)

            self.step_update()
            sampled_spins = self.symplectic_integrator.sample_spins()
            if use_window and self.__go_sampling:
                self.window.update(sampled_spins)

            self.check_stop(use_window)

        return sampled_spins

    def heat(self, position_stash: torch.Tensor) -> None:
        torch.add(self.symplectic_integrator.position, self.time_step * self.heat_parameter * position_stash, out=self.symplectic_integrator.position)

    def compute_symplectic_coefficients(self) -> Tuple[float, float, float]:
        pressure = self.pressure
        momentum_coefficient = self.time_step * (1. + pressure)
        position_coefficient = self.time_step * (pressure - 1.)
        quadratic_coefficient = self.time_step * self.xi0
        return momentum_coefficient, position_coefficient, quadratic_coefficient

    @property
    def pressure(self):
        return minimum(self.time_step * self.step * self.pressure_slope, 1.)
    
    def get_final_spins(self, spins: torch.Tensor, use_window: bool) -> torch.Tensor:
        if use_window: 
            if self.window.has_bifurcated_spins():
                return self.window.get_bifurcated_spins()
            else:
                print('No agent bifurcated. Returned final momentums\' signs instead.')
                return spins
        else:
            return spins