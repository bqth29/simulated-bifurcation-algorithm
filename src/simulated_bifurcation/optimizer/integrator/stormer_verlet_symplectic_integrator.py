from typing import Callable, Tuple

import torch
from numpy import minimum

from .abc_symplectic_integrator import ABCSymplecticIntegrator


class StormerVerletSymplecticIntegrator(ABCSymplecticIntegrator):
    """
    Order-2 symplectic integrator based on the Störmer-Verlet integration method.
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

    def integrate(self):
        intermediate_position = self.position.clone()
        intermediate_momentum = self.momentum.clone()
        n = 4
        for _ in range(n):
            auxiliary_momentum = torch.add(
                intermediate_momentum,
                intermediate_position,
                alpha=self.time_step * (self.get_current_pressure() - 1.0) / (2.0 * n),
            )
            auxiliary_momentum = torch.addmm(
                auxiliary_momentum,
                self.quadratic_tensor,
                self.activation_function(intermediate_position),
                alpha=self.time_step * self.quadratic_scale_parameter / (2.0 * n),
            )
            torch.add(
                intermediate_position,
                intermediate_momentum,
                alpha=self.time_step / n,
                out=intermediate_position,
            )
            torch.add(
                auxiliary_momentum,
                intermediate_position,
                alpha=self.time_step * (self.get_current_pressure() - 1.0) / (2.0 * n),
                out=intermediate_momentum,
            )
            intermediate_momentum = torch.addmm(
                intermediate_momentum,
                self.quadratic_tensor,
                self.activation_function(intermediate_position),
                alpha=self.time_step * self.quadratic_scale_parameter / (2.0 * n),
            )
        self.position = intermediate_position.clone()
        self.momentum = intermediate_momentum.clone()
