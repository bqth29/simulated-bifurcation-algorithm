import simulated_bifurcation as sb
import numpy as np

class Knapsack(sb.SBModel):

    """
    This class builds a SB solver for the integer-weights knapsack problem.
    """

    def __init__(self, weights: list, costs: list, max_weight: int) -> None:
        super().__init__()
        self.weights = np.array(weights)
        self.n_weights = len(weights)
        self.costs = np.array(costs)
        self.max_weight = max_weight
        self.dimension = self.n_weights + self.max_weight
        self.to_keep = None

    @property
    def weight_load(self) -> int:
        if self.to_keep is None: return 0
        else: return np.sum(self.weights[self.to_keep])

    @property
    def total_cost(self) -> float:
        if self.to_keep is None: return 0
        else: return np.sum(self.costs[self.to_keep])

    def __to_Ising__(self) -> sb.Ising:

        integer_array = np.arange(1, self.max_weight + 1)

        integer_matrix = integer_array.reshape(-1, 1) @ integer_array.reshape(1, -1)
        weights_matrix = self.weights.reshape(-1, 1) @ self.weights.reshape(1, -1)
        weights_integer_matrix = self.weights.reshape(-1, 1) @ integer_array.reshape(1, -1)

        costs_sum = np.sum(self.costs)
        max_cost = np.max(self.costs)
        weights_sum = np.sum(self.weights)

        A = 1.1 * max_cost#costs_sum

        J = A * np.block(
            [
                [-.5 * weights_matrix, weights_integer_matrix],
                [weights_integer_matrix.T, -.5 * (integer_matrix + np.ones((self.max_weight, self.max_weight)))]
            ]
        )

        J -= np.diag(np.diag(J))

        weights_factor = self.max_weight * (self.max_weight + 1) / 4 - weights_sum / 2

        h = np.block(
            [
                [A * weights_factor * (-1.) * self.weights.reshape(-1, 1) - self.costs.reshape(-1, 1) / 2],
                [A * weights_factor * integer_array.reshape(-1, 1) - A * (1 - self.max_weight / 2) * np.ones((self.max_weight, 1))]
            ]
        )

        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        print(ising)
        self.to_keep = (ising.ground_state.reshape(-1,) == 1)[:self.n_weights]