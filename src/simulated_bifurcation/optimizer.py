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


class Optimizer():

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

        self.agents_progress = tqdm(total=agents, desc='Bifurcated agents',
                                    disable=not verbose, smoothing=0)
        self.iterations_progress = tqdm(total=max_steps, desc='Iterations',
                                        disable=not verbose, smoothing=0.1,
                                        mininterval=0.5)

        # Optimizer setting
        self.mode = OptimizerMode.BALLISTIC if ballistic else OptimizerMode.DISCRETE

        self.heat_parameter = heat_parameter
        self.heated = heat_parameter is not None

        # Simulation parameters
        self.initialized = False
        self.time_step = time_step
        self.agents = agents

        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps

        # Quantum parameters
        self.gerschgorin = gerschgorin
        self.pressure = lambda t: minimum(
                pressure_slope * t, 1.)

        # Evolutive parameters
        self.X, self.Y = None, None
        self.dimension = None
        self.ising_model = None
        self.current_spins = None
        self.final_spins = None
        self.stability = None
        self.bifurcated = None
        self.previously_bifurcated = None
        self.new_bifurcated = None
        self.equal = None
        self.run = True
        self.xi0 = None
        self.step = 0
        self.time = 0

    @property
    def shape(self) -> Tuple[Union[int, None], int]:
        if self.initialized: return (self.dimension, self.agents)
        else: return (None, self.agents)

    def set_ballistic(self) -> None:
        if self.ballistic: pass
        else:
            self.discrete = False
            self.ballistic = True
            self.mode = OptimizerMode.BALLISTIC
    
    def set_discrete(self) -> None:
        if self.discrete: pass
        else:
            self.discrete = True
            self.ballistic = False
            self.mode = OptimizerMode.DISCRETE

    def confine(self) -> None:
        self.Y[torch.abs(self.X) > 1.] = 0
        torch.clip(self.X, -1., 1., out=self.X)

    def update_window(self) -> None:
        torch.eq(torch.einsum('ik, ik -> k', self.current_spins, torch.sign(self.X)),
                 self.dimension, out=self.equal)
        not_bifurcated = torch.logical_not(self.bifurcated)
        not_equal = torch.logical_not(self.equal)
        self.stability[torch.logical_and(self.equal, not_bifurcated)] += 1
        self.stability[torch.logical_and(not_equal, not_bifurcated)] = 0

        torch.eq(self.stability, self.convergence_threshold - 1,
                 out=self.bifurcated)

        torch.logical_xor(self.bifurcated, self.previously_bifurcated,
                       out=self.new_bifurcated)

        self.previously_bifurcated = self.bifurcated * 1.

        self.final_spins[:, self.new_bifurcated] = torch.sign(
            self.X[:, self.new_bifurcated])

        torch.sign(self.X, out=self.current_spins)

        self.agents_progress.update(self.new_bifurcated.sum().item())

    def reset(self, matrix: torch.Tensor) -> None:
        self.dimension = matrix.shape[0]
        self.ising_model = matrix

        self.X = 2 * torch.rand(size=(self.dimension, self.agents), 
                device = matrix.device, dtype=matrix.dtype) - 1
        self.Y = 2 * torch.rand(size=(self.dimension, self.agents),
                device = matrix.device, dtype=matrix.dtype) - 1

        # Stopping window

        self.current_spins = torch.zeros((self.dimension, self.agents),
                device = matrix.device, dtype=matrix.dtype)
        self.final_spins = torch.zeros((self.dimension, self.agents),
                device = matrix.device, dtype=matrix.dtype)

        self.stability = torch.zeros(self.agents,
                device = matrix.device, dtype=matrix.dtype)
        self.new_bifurcated = torch.zeros(self.agents, dtype=bool,
                device = matrix.device)
        self.previously_bifurcated = torch.zeros(self.agents, dtype=bool,
                device = matrix.device)
        self.bifurcated = torch.zeros(self.agents, dtype=bool,
                device = matrix.device)
        self.equal = torch.zeros(self.agents, dtype=bool,
                device = matrix.device)

        self.run = True
        self.step = 0

        if not self.gerschgorin:
            self.xi0 = 0.7 / \
                (torch.std(self.ising_model) * (self.dimension)**(1/2))
        else:
            self.xi0 = 1. / torch.max(
                torch.sum(torch.abs(self.ising_model), axis=1))

        self.initialized = True

    def step_update(self) -> None:
        self.step += 1
        self.iterations_progress.update()

    def update_X(self) -> None:
        pressure = self.pressure(self.time_step * self.step)
        torch.add(self.X, self.time_step * (1. + pressure) * self.Y, out=self.X)
    
    def update_Y(self) -> None:
        pressure = self.pressure(self.time_step * self.step)
        torch.add(
            self.Y,
            self.time_step * (pressure - 1.) * self.X,
            out=self.Y
        )

    def quadratic_update(self) -> None:
        temp = self.ising_model @ self.mode(self.X)
        torch.add(
            self.Y,
            self.time_step * self.xi0 * temp,
            out=self.Y
        )

    def check_stop(self, use_window: bool) -> None:
        if use_window and self.step % self.sampling_period == 0:
            self.update_window()
            self.run = torch.any(self.stability < self.convergence_threshold - 1)

        if self.step >= self.max_steps:
            self.run = False

    def symplectic_update(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:
        
        self.reset(matrix)

        while self.run:

            if self.heated: heatY = self.Y.clone().detach()

            self.update_Y()
            self.update_X()
            self.quadratic_update()
            self.confine()

            if self.heated: torch.add(self.Y, self.time_step * self.heat_parameter * heatY, out=self.Y)

            self.step_update()
            self.check_stop(use_window)

        self.agents_progress.close()
        self.iterations_progress.close()

        if use_window: 
            if torch.any(self.bifurcated):
                return self.final_spins[:, self.bifurcated]
            else:
                print('No agent bifurcated. Returned final oscillators instead.')
                return torch.sign(self.X)
        else: return torch.sign(self.X)