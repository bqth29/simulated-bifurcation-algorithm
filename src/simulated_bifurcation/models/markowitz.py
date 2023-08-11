from typing import Optional, Union

import numpy as np
import torch

from ..polynomial import IntegerPolynomial


class Markowitz(IntegerPolynomial):

    """
    A representation of the Markowitz model for portfolio optimization.
    Portfolio only takes integer stocks.
    """

    def __init__(
        self,
        covariance: Union[torch.Tensor, np.ndarray],
        expected_return: Union[torch.Tensor, np.ndarray],
        risk_coefficient: float = 1,
        number_of_bits: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        # Data
        super().__init__(
            -risk_coefficient * covariance,
            -expected_return,
            None,
            number_of_bits,
            dtype,
            device,
        )
        self.covariance = - self.matrix / risk_coefficient
        self.expected_return = - self.vector
        self.risk_coefficient = risk_coefficient

    @property
    def portfolio(self) -> Optional[torch.Tensor]:
        return self.sb_result[
            :, torch.argmin(self(self.sb_result.t())).item()
        ] if self.sb_result is not None else None

    @property
    def gains(self) -> float:
        return -self(self.portfolio) if self.portfolio is not None else 0.0
