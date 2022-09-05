import simulated_bifurcation as sb
import numpy as np

class QUBO(sb.SBModel):

    """
    Simulated bifurcation-based solver for the Quadratic unconstrained binary optimization (QUBO).

    Solving a QUBO for an upper trianglular matrix Q means searching the binary vector X
    such that the quantity `ΣΣ Q(i,j)x(i)x(j)` is minimal.
    """

    def __init__(self, Q: np.ndarray) -> None:
        super().__init__()
        self.Q = Q
        self.X = None

    def __len__(self): return self.Q.shape[0]

    @property
    def objective_function(self) -> float: return (self.X.T @ self.Q @self.X)[0, 0]

    def __to_Ising__(self) -> sb.Ising:
        
        J = -4 * (self.Q + self.Q.T)
        J -= np.diag(np.diag(J))
        h = -4 * self.Q @ np.ones((len(self), 1))

        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None: self.X = .5 * (ising.ground_state + 1)