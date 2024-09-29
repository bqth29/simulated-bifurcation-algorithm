from typing import Callable, Tuple

import torch


class SymplecticIntegrator:
    """
    Simulates the evolution of spins' momentum and position following
    the Hamiltonian quantum mechanics equations that drive the
    Simulated Bifurcation (SB) algorithm.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.position = self.__init_oscillator(shape, dtype, device)
        self.momentum = self.__init_oscillator(shape, dtype, device)
        self.activation_function = activation_function

    @staticmethod
    def __init_oscillator(
        shape: Tuple[int, int], dtype: torch.dtype, device: torch.device
    ):
        return 2 * torch.rand(size=shape, device=device, dtype=dtype) - 1

    def position_update(self, coefficient: float) -> None:
        torch.add(self.position, self.momentum, alpha=coefficient, out=self.position)

    def momentum_update(self, coefficient: float) -> None:
        torch.add(self.momentum, self.position, alpha=coefficient, out=self.momentum)

    def quadratic_momentum_update(
        self, coefficient: float, matrix: torch.Tensor
    ) -> None:
        # do not use out=self.position because of side effects
        self.momentum = torch.addmm(
            self.momentum,
            matrix,
            self.activation_function(self.position),
            alpha=coefficient,
        )

    def simulate_inelastic_walls(self) -> None:
        self.momentum[torch.abs(self.position) > 1.0] = 0.0
        torch.clip(self.position, -1.0, 1.0, out=self.position)

    def step(
        self,
        momentum_coefficient: float,
        position_coefficient: float,
        quadratic_coefficient: float,
        matrix: torch.Tensor,
    ) -> None:
        self.momentum_update(momentum_coefficient)
        self.position_update(position_coefficient)
        self.quadratic_momentum_update(quadratic_coefficient, matrix)
        self.simulate_inelastic_walls()

    def sample_spins(self) -> torch.Tensor:
        return torch.where(self.position >= 0.0, 1.0, -1.0)
