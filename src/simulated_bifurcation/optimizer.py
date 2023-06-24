from typing import Tuple
import torch
from tqdm import tqdm
from numpy import minimum
from enum import Enum


PRESSURE_SLOPE = .01
HEAT_COEFFICIENT = .06


class OptimizerMode(Enum):
    BALLISTIC = torch.nn.Identity()
    DISCRETE = torch.sign

    @property
    def activation_function(self) -> str: return self.value


class SymplecticIntegrator:

    def __init__(self, shape: Tuple[int, int], mode: OptimizerMode, dtype: torch.dtype, device: str):
        self.momentum = SymplecticIntegrator.__init_oscillator(shape, dtype, device)
        self.position = SymplecticIntegrator.__init_oscillator(shape, dtype, device)
        self.activation_function = mode.activation_function

    @staticmethod
    def __init_oscillator(shape: Tuple[int, int], dtype: torch.dtype, device: str):
        return 2 * torch.rand(size=shape, device=device, dtype=dtype) - 1

    def __momentum_update(self, coefficient: float) -> None:
        torch.add(self.momentum, coefficient * self.position, out=self.momentum)
    
    def __position_update(self, coefficient: float) -> None:
        torch.add(self.position, coefficient * self.momentum, out=self.position)

    def __quadratic_position_update(self, coefficient: float, matrix: torch.Tensor) -> None:
        torch.add(self.position, coefficient * matrix @ self.activation_function(self.momentum), out=self.position)

    def __simulate_inelastic_walls(self) -> None:
        self.position[torch.abs(self.momentum) > 1.] = 0
        torch.clip(self.momentum, -1., 1., out=self.momentum)

    def step(self, momentum_coefficient: float, position_coefficient: float, quadratic_coefficient: float, matrix: torch.Tensor) -> None:
        self.__momentum_update(momentum_coefficient)
        self.__position_update(position_coefficient)
        self.__quadratic_position_update(quadratic_coefficient, matrix)
        self.__simulate_inelastic_walls()

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
            smoothing=0,
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

    def __update_final_spins(self, sampled_spins) -> None:
        self.final_spins[:, self.newly_bifurcated] = torch.sign(
            sampled_spins[:, self.newly_bifurcated]
        )

    def __set_previously_bifurcated_spins(self) -> None:
        self.previously_bifurcated = self.bifurcated * 1.

    def __set_newly_bifurcated_spins(self) -> None:
        torch.logical_xor(
            self.bifurcated,
            self.previously_bifurcated,
            out=self.newly_bifurcated
        )

    def __update_bifurcated_spins(self) -> None:
        torch.eq(
            self.stability,
            self.convergence_threshold - 1,
            out=self.bifurcated
        )

    def __update_stability_streak(self) -> None:
        self.stability[torch.logical_and(self.equal, self.not_bifurcated)] += 1
        self.stability[torch.logical_and(self.not_equal, self.not_bifurcated)] = 0

    @property
    def not_equal(self) -> torch.Tensor:
        return torch.logical_not(self.equal)

    @property
    def not_bifurcated(self) -> torch.Tensor:
        return torch.logical_not(self.bifurcated)

    def __compare_spins(self, sampled_spins: torch.Tensor) -> None:
        torch.eq(
            torch.einsum('ik, ik -> k', self.current_spins, sampled_spins),
            self.n_spins,
            out=self.equal
        )

    def __store_spins(self, sampled_spins: torch.Tensor) -> None:
        torch.sign(sampled_spins, out=self.current_spins)

    def __get_number_newly_bifurcated_agents(self) -> int:
        return self.newly_bifurcated.sum().item()
    
    def update(self, sampled_spins: torch.Tensor):
        self.__compare_spins(sampled_spins)
        self.__update_stability_streak()
        self.__update_bifurcated_spins()
        self.__set_newly_bifurcated_spins()
        self.__set_previously_bifurcated_spins()
        self.__update_final_spins(sampled_spins)
        self.__store_spins(sampled_spins)
        self.progress.update(self.__get_number_newly_bifurcated_agents())
    
    def must_continue(self) -> bool:
        return torch.any(self.stability < self.convergence_threshold - 1)
    
    def has_bifurcated_spins(self) -> bool:
        return torch.any(self.bifurcated)

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
        ballistic: bool,
        heat: bool,
        verbose: bool
    ) -> None:
        # Optimizer setting
        self.mode = OptimizerMode.BALLISTIC if ballistic else OptimizerMode.DISCRETE
        self.window = None
        self.symplectic_integrator = None
        self.heat_parameter = HEAT_COEFFICIENT
        self.heated = heat
        self.verbose = verbose
        # Simulation parameters
        self.time_step = time_step
        self.agents = agents
        self.pressure_slope = PRESSURE_SLOPE
        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps

    def __reset(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.__init_progress_bar(self.max_steps, self.verbose)
        self.__init_symplectic_integrator(matrix)
        self.__init_window(matrix, use_window)
        self.__init_xi0(matrix)
        self.run = True
        self.step = 0

    def __init_progress_bar(self, max_steps: int, verbose: bool) -> None:
        self.iterations_progress = tqdm(
            total=max_steps, desc='Iterations',
            disable=not verbose, smoothing=0.1,
            mininterval=0.5
        )

    def __init_xi0(self, matrix: torch.Tensor):
        self.xi0 =  0.7 / (torch.std(matrix) * (matrix.shape[0])**(1/2))

    def __init_window(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.window = StopWindow(
            matrix.shape[0], self.agents,
            self.convergence_threshold, self.sampling_period,
            matrix.dtype, str(matrix.device), (self.verbose and use_window)
        )

    def __init_symplectic_integrator(self, matrix: torch.Tensor) -> None:
        self.symplectic_integrator = SymplecticIntegrator(
            (matrix.shape[0], self.agents),
            self.mode, matrix.dtype, str(matrix.device)
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

    def __symplectic_update(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:

        while self.run:

            if self.heated: position_copy = self.symplectic_integrator.position.clone().detach()

            momentum_coefficient, position_coefficient, quadratic_coefficient = self.__compute_symplectic_coefficients()
            self.symplectic_integrator.step(momentum_coefficient, position_coefficient, quadratic_coefficient, matrix)

            if self.heated: self.__heat(position_copy)

            self.__step_update()
            sampled_spins = self.symplectic_integrator.sample_spins()
            if use_window and self.__do_sampling:
                self.window.update(sampled_spins)

            self.__check_stop(use_window)

        return sampled_spins

    def __heat(self, position_stash: torch.Tensor) -> None:
        torch.add(self.symplectic_integrator.position, self.time_step * self.heat_parameter * position_stash, out=self.symplectic_integrator.position)

    def __compute_symplectic_coefficients(self) -> Tuple[float, float, float]:
        pressure = self.__pressure
        momentum_coefficient = self.time_step * (1. + pressure)
        position_coefficient = self.time_step * (pressure - 1.)
        quadratic_coefficient = self.time_step * self.xi0
        return momentum_coefficient, position_coefficient, quadratic_coefficient

    @property
    def __pressure(self):
        return minimum(self.time_step * self.step * self.pressure_slope, 1.)
    
    def run_integrator(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:
        self.__reset(matrix, use_window)
        spins = self.__symplectic_update(matrix, use_window)
        self.__close_progress_bars()
        return self.get_final_spins(spins, use_window)
    
    def get_final_spins(self, spins: torch.Tensor, use_window: bool) -> torch.Tensor:
        if use_window: 
            if self.window.has_bifurcated_spins():
                return self.window.get_bifurcated_spins()
            else:
                print('No agent bifurcated. Returned final momentums\' signs instead.')
                return spins
        else:
            return spins
