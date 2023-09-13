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
            -risk_coefficient / 2 * covariance,
            expected_return,
            None,
            number_of_bits,
            dtype,
            device,
        )
        self.risk_coefficient = risk_coefficient

    @property
    def covariance(self) -> torch.Tensor:
        return -(2 / self.risk_coefficient) * self.matrix

    @property
    def expected_return(self) -> torch.Tensor:
        return self.vector

    @property
    def portfolio(self) -> Optional[torch.Tensor]:
        return (
            self.sb_result[:, torch.argmax(self(self.sb_result.t())).item()]
            if self.sb_result is not None
            else None
        )

    @property
    def gains(self) -> float:
        return self(self.portfolio) if self.portfolio is not None else 0.0


class SequentialMarkowitz(IntegerPolynomial):
    def __init__(
        self,
        covariances: Union[torch.Tensor, np.ndarray],
        expected_returns: Union[torch.Tensor, np.ndarray],
        rebalancing_costs: Union[torch.Tensor, np.ndarray],
        initial_stocks: Optional[Union[torch.Tensor, np.ndarray]] = None,
        risk_coefficient: float = 1,
        number_of_bits: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """
        covariances : T x N x N
        expected_returns : T x N
        rebalancing_costs : T x N x N
        """
        self.covariances = covariances
        self.expected_returns = expected_returns
        self.rebalancing_costs = rebalancing_costs
        self.risk_coefficient = risk_coefficient
        self.timestamps = covariances.shape[0]
        self.dimension = covariances.shape[0] * covariances.shape[1]

        self.initial_stocks = (
            torch.zeros(self.dimension) if initial_stocks is None else initial_stocks
        )
