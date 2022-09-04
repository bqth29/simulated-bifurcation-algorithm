import simulated_bifurcation as sb
import numpy as np

class BinaryLinearProgramming(sb.SBModel):

    """
    Maximizes the value of `c.x` subjected to the constraint `Sx = b`, where `x` is a binary vector.
    """

    def __init__(self, S: np.ndarray, b: np.ndarray, c: np.ndarray, penalty_coefficient: float, objective_coefficient: float) -> None:

        """
        Parameters
        ----------

        S : numpy.ndarray
            Matrix of the equality constraint
        b : numpy.ndarray
            Vector of the equality constraint
        c : numpy.ndarray
            Vector of the equality to maximize
        penalty_coefficient : float
            Weight of the value to maximize with respect to the constraint
        """

        super().__init__()
        self.S = S
        self.b = b
        self.c = c

        self.penalty_coefficient = penalty_coefficient
        self.objective_coefficient = objective_coefficient

        if penalty_coefficient / objective_coefficient < c.shape[0] * max(c.max(), 0): print(
            "WARNING: The ratio between the penalty and the objective coefficients "
            "cannot ensure that the constraint will be verified!"
        )

        self.value = None
        self.solution = None
        self.acceptable = None

    def __to_Ising__(self) -> sb.Ising:

        dimension = self.c.shape[0] 

        J_binary = -2 * self.penalty_coefficient * self.S.T @ self.S
        h_binary = -2 * self.penalty_coefficient * np.einsum('ik, il -> lk', self.b, self.S) - self.objective_coefficient * self.c

        J = .25 * J_binary
        h = .5 * h_binary - .25 * J_binary @ np.ones((dimension, 1))

        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        
        self.solution = .5 * (1 + ising.ground_state)
        self.acceptable = np.all(self.S @ self.solution == self.b)
        self.value = np.dot(self.solution.reshape(-1,), self.c.reshape(-1,))