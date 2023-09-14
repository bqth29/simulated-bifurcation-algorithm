from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..polynomial import IntegerPolynomial
from .utils import cast_to_tensor


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
        if self.sb_result is None:
            return None
        best_agent = torch.argmax(self(self.sb_result.t())).item()
        return self.sb_result[:, best_agent]

    @property
    def gains(self) -> float:
        return self(self.portfolio).item() if self.portfolio is not None else 0.0


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
        self.covariances = cast_to_tensor(covariances)
        self.expected_returns = cast_to_tensor(expected_returns)
        self.rebalancing_costs = cast_to_tensor(rebalancing_costs)
        self.risk_coefficient = risk_coefficient
        self.timestamps = self.covariances.shape[0]
        self.assets = self.covariances.shape[1]

        self.initial_stocks = (
            torch.zeros(self.assets) if initial_stocks is None else initial_stocks
        )

        matrix, vector, constant = self.compile_model()
        super().__init__(matrix, vector, constant, number_of_bits, dtype, device)

    def compile_model(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        matrix = self.__compile_matrix()
        vector = self.__compile_vector()
        constant = self.__compile_constant()
        return matrix, vector, constant

    def __compile_matrix(self) -> torch.Tensor:
        block_diagonal = self.__create_block_diagonal()
        upper_block_diagonal = self.__create_upper_block_diagonal()
        return block_diagonal + upper_block_diagonal + upper_block_diagonal.t()

    def __create_block_diagonal(self) -> torch.Tensor:
        time_shifted_rebalancing_costs = torch.roll(self.rebalancing_costs, -1, 0)
        time_shifted_rebalancing_costs[-1] = 0.0
        tensors = (
            -self.risk_coefficient / 2 * self.covariances
            - self.rebalancing_costs
            - time_shifted_rebalancing_costs
        )
        return torch.block_diag(*tensors)

    def __create_upper_block_diagonal(self) -> torch.Tensor:
        dimension = self.assets * self.timestamps
        tensor = torch.zeros(dimension, dimension)
        upper_block_diagonal = torch.block_diag(*self.rebalancing_costs[1:])
        tensor[: -self.assets, self.assets :] = upper_block_diagonal
        return tensor

    def __compile_vector(self) -> torch.Tensor:
        stacked_expected_returns = self.expected_returns.reshape(-1, 1)
        first_timestamp_rebalancing_costs = self.rebalancing_costs[0]
        initial_stocks_contribution = (
            first_timestamp_rebalancing_costs + first_timestamp_rebalancing_costs.t()
        ) @ self.initial_stocks.reshape(-1, 1)
        stacked_expected_returns[: self.assets] += initial_stocks_contribution
        return stacked_expected_returns

    def __compile_constant(self) -> float:
        initial_portfolio = self.initial_stocks.reshape(-1, 1)
        constant = (
            -initial_portfolio.t() @ self.rebalancing_costs[0] @ initial_portfolio
        )
        return round(constant.item(), 4)

    @property
    def portfolio(self) -> torch.Tensor:
        if self.sb_result is None:
            return None
        best_agent = torch.argmax(self(self.sb_result.t())).item()
        return self.sb_result[:, best_agent].reshape(self.timestamps, self.assets)

    @property
    def gains(self) -> float:
        return (
            self(self.portfolio.reshape(1, -1)).item()
            if self.portfolio is not None
            else 0.0
        )
