from typing import Callable, Tuple

import torch

from ..core.tensor_bearer import TensorBearer


class SymplecticIntegrator(TensorBearer):
    """
    Simulates the evolution of spins' momentum and position following
    the Hamiltonian quantum mechanics equations that drive the
    Simulated Bifurcation (SB) algorithm.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        time_step: float,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        heat: bool,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(dtype=dtype, device=device)
        self.position = self.__init_oscillator(shape, dtype, device)
        self.momentum = self.__init_oscillator(shape, dtype, device)
        self.time_step = time_step
        self.activation_function = activation_function
        self.heat = heat

    @staticmethod
    def __init_oscillator(
        shape: Tuple[int, int], dtype: torch.dtype, device: torch.device
    ):
        return 2.0 * torch.rand(size=shape, device=device, dtype=dtype) - 1.0

    def position_update(self) -> None:
        torch.add(
            self.position,
            self.momentum,
            alpha=self.time_step,
            out=self.position,
        )

    def momentum_update(self, coefficient: float) -> None:
        torch.add(
            self.momentum,
            self.position,
            alpha=self.time_step * coefficient,
            out=self.momentum,
        )

    def quadratic_momentum_update(
        self, coefficient: float, matrix: torch.Tensor
    ) -> None:
        # do not use out=self.position because of side effects
        self.momentum = torch.addmm(
            self.momentum,
            matrix,
            self.activation_function(self.position),
            alpha=self.time_step * coefficient,
        )

    def simulate_inelastic_walls(self) -> None:
        self.momentum[torch.abs(self.position) > 1.0] = 0.0
        torch.clip(self.position, -1.0, 1.0, out=self.position)

    def simulate_heating(
        self, momentum_copy: torch.Tensor, heat_coefficient: float
    ) -> None:
        torch.add(
            self.momentum,
            momentum_copy,
            alpha=self.time_step * heat_coefficient,
            out=self.momentum,
        )

    def step(
        self,
        momentum_coefficient: float,
        quadratic_coefficient: float,
        heat_coefficient: float,
        matrix: torch.Tensor,
    ) -> None:
        if self.heat:
            momentum_copy = self.momentum.clone()
        self.momentum_update(momentum_coefficient)
        self.position_update()
        self.quadratic_momentum_update(quadratic_coefficient, matrix)
        self.simulate_inelastic_walls()
        if self.heat:
            self.simulate_heating(momentum_copy, heat_coefficient)

    def sample_spins(self) -> torch.Tensor:
        return torch.where(self.position >= 0.0, 1.0, -1.0)
