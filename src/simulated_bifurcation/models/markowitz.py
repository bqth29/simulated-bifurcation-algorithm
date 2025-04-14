from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..core.quadratic_polynomial import QuadraticPolynomial
from .abc_quadratic_model import ABCQuadraticModel


class SequentialMarkowitz(ABCQuadraticModel):
    """
    Implementation of the Markowitz model for the integer
    trading trajectory optimization problem.

    In finance, the trading trajectory problem involves identifying
    the most effective combination of portfolios and associated trading
    actions that maximize future expected returns during a specific time
    frame, considering both transaction costs and anticipated risks.

    Mathematically, this problem can be written as:

    .. math::

        \\hbox{argmax } \\sum_{t = 1}^{T} w_{t}^{T} \\mu_{t} - \\frac{\\gamma}{2}
        w_{t}^{T} \\Sigma_{t} w_{t} - \\Delta w_t^T \\Lambda_t \\Delta w_t

    where:

    - :math:`1, ..., t, ... T` are the trading decision instants
    - :math:`w_{t}` is the stocks portfolio at time :math:`t`. All the stocks
      are integer values.
    - :math:`\\Sigma_{t}` is the assets' covariance matrix at time :math:`t`
    - :math:`\\mu_{t}` is the vector of expected returns at time :math:`t`
    - :math:`\\gamma` is a risk aversion coefficient
    - :math:`\\Lambda_{t} = w_{t} - w_{t - 1}` are the rebalancing transaction costs
      resulting from the stock difference between time :math:`t - 1` and :math:`t`
    """

    sense = "maximize"

    def __init__(
        self,
        covariances: Union[torch.Tensor, np.ndarray],
        expected_returns: Union[torch.Tensor, np.ndarray],
        rebalancing_costs: Union[torch.Tensor, np.ndarray],
        initial_stocks: Optional[Union[torch.Tensor, np.ndarray]] = None,
        risk_coefficient: float = 1,
        number_of_bits: int = 1,
    ) -> None:
        """
        Instantiates a Markowitz problem model that can be optimized using
        the Simulated Bifurcation algorithm.
        """
        self.covariances = covariances
        self.expected_returns = expected_returns
        self.rebalancing_costs = rebalancing_costs
        self.risk_coefficient = risk_coefficient
        self.timestamps = covariances.shape[0]
        self.assets = covariances.shape[1]

        self.initial_stocks = (
            torch.zeros(
                self.assets,
                dtype=rebalancing_costs.dtype,
                device=rebalancing_costs.device,
            )
            if initial_stocks is None
            else initial_stocks
        )

        self.number_of_bits = number_of_bits
        self.domain = f"int{number_of_bits}"

        self.shape = self.timestamps, self.assets

    def _as_quadratic_polynomial(
        self, dtype: Optional[torch.dtype], device: Optional[Union[str, torch.device]]
    ) -> QuadraticPolynomial:
        return QuadraticPolynomial(
            self.__compile_matrix(),
            self.__compile_vector(),
            self.__compile_offset(),
            dtype=dtype,
            device=device,
        )

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
        return stacked_expected_returns.reshape(
            -1,
        )

    def __compile_offset(self) -> float:
        initial_portfolio = self.initial_stocks.reshape(-1, 1)
        constant = -initial_portfolio.T @ self.rebalancing_costs[0] @ initial_portfolio
        return float(constant)

    def _from_optimized_tensor(
        self, optimized_tensor: torch.Tensor, optimized_cost: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        return optimized_tensor.reshape(*self.shape), optimized_cost.item()


class Markowitz(SequentialMarkowitz):
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
    ) -> None:
        covariance = torch.unsqueeze(covariance, 0)
        expected_return = torch.unsqueeze(expected_return, 0)
        rebalancing_costs = torch.zeros_like(covariance)
        super().__init__(
            covariance,
            expected_return,
            rebalancing_costs,
            None,
            risk_coefficient,
            number_of_bits,
        )
        self.shape = (-1,)


def markowitz(
    covariance: Union[torch.Tensor, np.ndarray],
    expected_return: Union[torch.Tensor, np.ndarray],
    risk_coefficient: float = 1,
    number_of_bits: int = 1,
) -> Markowitz:
    return Markowitz(covariance, expected_return, risk_coefficient, number_of_bits)


def sequential_markowitz(
    covariances: Union[torch.Tensor, np.ndarray],
    expected_returns: Union[torch.Tensor, np.ndarray],
    rebalancing_costs: Union[torch.Tensor, np.ndarray],
    initial_stocks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    risk_coefficient: float = 1,
    number_of_bits: int = 1,
) -> SequentialMarkowitz:
    return SequentialMarkowitz(
        covariances,
        expected_returns,
        rebalancing_costs,
        initial_stocks,
        risk_coefficient,
        number_of_bits,
    )
