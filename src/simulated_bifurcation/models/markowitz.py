from .integer import Integer
import torch
from typing import Union

class Markowitz(Integer):

    """
    A representation of the Markowitz model for portolio optimization.
    Portfolio only takes integer stocks.
    """

    def __init__(
        self, 
        covariance: torch.Tensor, 
        expected_return: torch.Tensor, 
        risk_coefficient: float = 1, 
        number_of_bits: int = 1,
        dtype: torch.dtype=torch.float32,
        device: str = 'cpu'
    ) -> None:

        # Data
        self.covariance       = covariance.to(dtype=dtype, device=device)
        self.expected_return  = expected_return.to(dtype=dtype, device=device)
        self.risk_coefficient = risk_coefficient
        super().__init__(- risk_coefficient * covariance, 
                        - expected_return, number_of_bits,
                        dtype, device)

    @property
    def portfolio(self) -> torch.Tensor: return self.solution

    @property
    def objective_value(self) -> Union[float, None]:
        return - super().objective_value