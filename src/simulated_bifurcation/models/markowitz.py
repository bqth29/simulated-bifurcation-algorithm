from typing import Optional, Tuple, Union

import numpy as np
import torch

from .abc_model import ABCModel


class SequentialMarkowitz(ABCModel):
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
    - :math:`w_{t}` is the stocks portolio at time :math:`t`. All the stocks
      are integer values.
    - :math:`\\Sigma_{t}` is the assets' covariance matrix at time :math:`t`
    - :math:`\\mu_{t}` is the vector of expected returns at time :math:`t`
    - :math:`\\gamma` is a risk aversion coefficient
    - :math:`\\Lambda_{t} = w_{t} - w_{t - 1}` are the rebalancing transaction costs
      resulting from the stock difference between time :math:`t - 1` and :math:`t`
    """

    def __init__(
        self,
        covariances: Union[torch.Tensor, np.ndarray],
        expected_returns: Union[torch.Tensor, np.ndarray],
        rebalancing_costs: Union[torch.Tensor, np.ndarray],
        initial_stocks: Optional[Union[torch.Tensor, np.ndarray]] = None,
        risk_coefficient: float = 1,
        number_of_bits: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Instantiates a Markowitz problem model that can be optimized using
        the Simulated Bifurcation algorithm.
        """
        self.covariances = self.__cast_to_tensor(
            covariances, dtype=dtype, device=device
        )
        self.expected_returns = self.__cast_to_tensor(
            expected_returns, dtype=dtype, device=device
        )
        self.rebalancing_costs = self.__cast_to_tensor(
            rebalancing_costs, dtype=dtype, device=device
        )
        self.risk_coefficient = risk_coefficient
        self.timestamps = covariances.shape[0]
        self.assets = covariances.shape[1]

        self.initial_stocks = (
            torch.zeros(self.assets) if initial_stocks is None else initial_stocks
        )

        self.number_of_bits = number_of_bits
        self.domain = f"int{number_of_bits}"

        matrix, vector, constant = self.compile_model()
        super().__init__(
            matrix,
            vector.reshape(
                -1,
            ),
            constant,
            dtype=dtype,
            device=device,
        )

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
        return constant.item()

    def __cast_to_tensor(
        self,
        tensor_like: Union[torch.Tensor, np.ndarray],
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.to(dtype=dtype, device=device)
        else:
            return torch.from_numpy(tensor_like).to(dtype=dtype, device=device)

    @property
    def portfolio(self) -> Optional[torch.Tensor]:
        if self.sb_result is None:
            return None
        best_agent = torch.argmax(self(self.sb_result.t())).item()
        return self.sb_result[:, best_agent].reshape(self.timestamps, self.assets)

    @property
    def gains(self) -> float:
        return (
            self(self.portfolio.reshape(1, -1)).item()
            if self.portfolio is not None
            else None
        )


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
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
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
            dtype,
            device,
        )

    @property
    def covariance(self) -> torch.Tensor:
        return -(2 / self.risk_coefficient) * self._quadratic_coefficients

    @property
    def expected_return(self) -> torch.Tensor:
        return self._linear_coefficients

    @property
    def portfolio(self) -> Optional[torch.Tensor]:
        portfolio = super().portfolio
        return (
            portfolio.reshape(
                -1,
            )
            if portfolio is not None
            else None
        )
