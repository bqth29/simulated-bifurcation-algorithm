import torch
from typing import Tuple
from .optimizer_mode import OptimizerMode


class SymplecticIntegrator:

    """
    Simulates the evolution of spins' momentum and position following
    the Hamiltonian quantum mechanics equations that drive the 
    Simulated Bifurcation (SB) algorithm.
    """

    def __init__(self, shape: Tuple[int, int], mode: OptimizerMode, dtype: torch.dtype, device: str):
        self.momentum = SymplecticIntegrator.__init_oscillator(shape, dtype, device)
        self.position = SymplecticIntegrator.__init_oscillator(shape, dtype, device)
        self.activation_function = mode.activation_function

    @staticmethod
    def __init_oscillator(shape: Tuple[int, int], dtype: torch.dtype, device: str):
        return 2 * torch.rand(size=shape, device=device, dtype=dtype) - 1

    def momentum_update(self, coefficient: float) -> None:
        torch.add(self.momentum, coefficient * self.position, out=self.momentum)
    
    def position_update(self, coefficient: float) -> None:
        torch.add(self.position, coefficient * self.momentum, out=self.position)

    def quadratic_position_update(self, coefficient: float, matrix: torch.Tensor) -> None:
        torch.add(self.position, coefficient * matrix @ self.activation_function(self.momentum), out=self.position)

    def simulate_inelastic_walls(self) -> None:
        self.position[torch.abs(self.momentum) > 1.] = 0
        torch.clip(self.momentum, -1., 1., out=self.momentum)

    def step(self, momentum_coefficient: float, position_coefficient: float, quadratic_coefficient: float, matrix: torch.Tensor) -> None:
        self.momentum_update(momentum_coefficient)
        self.position_update(position_coefficient)
        self.quadratic_position_update(quadratic_coefficient, matrix)
        self.simulate_inelastic_walls()

    def sample_spins(self) -> torch.Tensor:
        return torch.sign(self.momentum)
