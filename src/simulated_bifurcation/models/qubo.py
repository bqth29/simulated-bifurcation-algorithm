from .binary import Binary
import torch


class QUBO(Binary):

    """
    Simulated bifurcation-based solver for the Quadratic unconstrained binary optimization (QUBO) problem.

    Solving a QUBO for an upper trianglular matrix Q means searching the binary vector X
    such that the quantity `ΣΣ Q(i,j)x(i)x(j)` is minimal.
    """

    def __init__(self, Q: torch.Tensor, dtype: torch.dtype=torch.float32,
                device: str = 'cpu') -> None:
        self.Q = Q.to(dtype=dtype, device=device)
        super().__init__(- (self.Q + self.Q.T), None, dtype, device)