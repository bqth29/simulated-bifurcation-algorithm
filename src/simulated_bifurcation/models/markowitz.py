from ..polynomial import IntegerPolynomial
import torch
import numpy as np
from typing import Union

class Markowitz(IntegerPolynomial):

    """
    A representation of the Markowitz model for portolio optimization.
    Portfolio only takes integer stocks.
    """

    def __init__(
        self, 
        covariance: Union[torch.Tensor, np.ndarray], 
        expected_return: Union[torch.Tensor, np.ndarray], 
        risk_coefficient: float = 1, 
        number_of_bits: int = 1,
        dtype: torch.dtype=torch.float32,
        device: str = 'cpu'
    ) -> None:

        # Data
        self.covariance = covariance.to(dtype=dtype, device=device)
        self.expected_return = expected_return.to(dtype=dtype, device=device)
        self.risk_coefficient = risk_coefficient
        super().__init__(- risk_coefficient * covariance, - expected_return, None, number_of_bits, dtype, device)

    @property
    def portfolio(self) -> Union[torch.Tensor, None]:
        return self.sb_result
    
    @property
    def gains(self) -> float:
        return - self(self.sb_result) if self.sb_result is not None else 0.