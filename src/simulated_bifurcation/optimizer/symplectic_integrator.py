from typing import Callable, Tuple

import torch
from numpy import minimum

from ..core.tensor_bearer import TensorBearer


class SymplecticIntegrator(TensorBearer):
    """
    Simulates the evolution of spins' momentum and position following the Hamiltonian quantum mechanics equations that
    drive the Simulated Bifurcation (SB) algorithm.
    """

    def __init__(
        self,
        n_oscillators: int,
        time_step: float,
        pressure_slope: float,
        heat_coefficient: float,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        heat: bool,
        quadratic_tensor: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(dtype=dtype, device=device)
        n_spins = quadratic_tensor.shape[0]
        self.position = self.init_oscillator((n_spins, n_oscillators))
        self.momentum = self.init_oscillator((n_spins, n_oscillators))
        self.time_step = time_step
        self.pressure_slope = pressure_slope
        self.heat_coefficient = heat_coefficient
        self.activation_function = activation_function
        self.heat = heat
        self.quadratic_tensor = self._cast_tensor(quadratic_tensor)
        self.quadratic_scale_parameter = (
            0.5 * (n_spins - 1) ** 0.5 / (torch.sqrt(torch.sum(quadratic_tensor**2)))
        )
        self.step = 0

    def init_oscillator(self, shape: Tuple[int, int]) -> torch.Tensor:
        return 2.0 * torch.rand(size=shape, device=self.device, dtype=self.dtype) - 1.0

    def position_update(self) -> None:
        torch.add(
            self.position,
            self.momentum,
            alpha=self.time_step,
            out=self.position,
        )

    def momentum_update(self) -> None:
        torch.add(
            self.momentum,
            self.position,
            alpha=self.time_step * (self.get_current_pressure() - 1.0),
            out=self.momentum,
        )

    def quadratic_momentum_update(self) -> None:
        # do not use out=self.position because of side effects
        self.momentum = torch.addmm(
            self.momentum,
            self.quadratic_tensor,
            self.activation_function(self.position),
            alpha=self.time_step * self.quadratic_scale_parameter,
        )

    def simulate_inelastic_walls(self) -> None:
        self.momentum[torch.abs(self.position) > 1.0] = 0.0
        torch.clip(self.position, -1.0, 1.0, out=self.position)

    def simulate_heating(self, momentum_copy: torch.Tensor) -> None:
        torch.add(
            self.momentum,
            momentum_copy,
            alpha=self.time_step * self.heat_coefficient,
            out=self.momentum,
        )

    def get_current_pressure(self) -> float:
        return minimum(self.time_step * self.step * self.pressure_slope, 1.0)

    def integration_step(self) -> None:
        if self.heat:
            momentum_copy = self.momentum.clone()
        self.momentum_update()
        self.quadratic_momentum_update()
        self.position_update()
        self.simulate_inelastic_walls()
        if self.heat:
            self.simulate_heating(momentum_copy)
        self.step += 1

    def sample_spins(self) -> torch.Tensor:
        return torch.where(self.position >= 0.0, 1.0, -1.0)
