from . import *


class QUBO(SBModel):

    """
    Simulated bifurcation-based solver for the Quadratic unconstrained binary optimization (QUBO) problem.

    Solving a QUBO for an upper trianglular matrix Q means searching the binary vector X
    such that the quantity `ΣΣ Q(i,j)x(i)x(j)` is minimal.
    """

    def __init__(self, Q: torch.Tensor,
                dtype: torch.dtype=torch.float32,
                device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.Q = Q.to(dtype=dtype, device=device)
        self.X = None

    def __len__(self): return self.Q.shape[0]

    @property
    def value(self) -> float: return (self.X.t() @ self.Q @ self.X).item()

    def __to_Ising__(self) -> Ising:
        
        J = - (self.Q + self.Q.T)
        h = (self.Q + self.Q.T) @ torch.ones((len(self), 1), device=self.device)

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None: self.X = .5 * (ising.ground_state + 1)