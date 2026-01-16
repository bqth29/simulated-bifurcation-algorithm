from typing import Callable

import torch

from .abc_symplectic_integrator import ABCSymplecticIntegrator


class EulerSymplecticIntegrator(ABCSymplecticIntegrator):
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
        super().__init__(
            n_oscillators=n_oscillators,
            time_step=time_step,
            pressure_slope=pressure_slope,
            heat_coefficient=heat_coefficient,
            activation_function=activation_function,
            heat=heat,
            quadratic_tensor=quadratic_tensor,
            dtype=dtype,
            device=device,
        )

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

    def integrate(self):
        self.momentum_update()
        self.quadratic_momentum_update()
        self.position_update()
