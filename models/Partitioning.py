import simulated_bifurcation as sb
import numpy as np

class NumberPartioning(sb.SBModel):

    """
    This class builds a SB solver to split a set of values in two subsets whose respective
    sum are as close as possible.
    """

    def __init__(self, set: list) -> None:
        super().__init__()
        self.set = set
        self.split = False
        self.partition = None

    def __len__(self): return len(self.set)

    def __str__(self) -> str:
        return f'Original set: {self.set}\nSubset 1: {self.partition[0]} (sum: {np.sum(self.partition[0])})\nSubset 2: {self.partition[1]} (sum: {np.sum(self.partition[1])})'

    def __to_Ising__(self) -> sb.Ising:
        
        J = -2 * np.array(self.set).reshape(-1, 1) @ np.array(self.set).reshape(1, -1)
        J -= np.diag(np.diag(J))
        h = np.zeros((len(self), 1))

        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        
        subset_1 = []
        subset_2 = []

        partition = ising.ground_state.reshape(-1,)

        for elt in range(len(self)):

            if partition[elt] > 0: subset_1.append(self.set[elt])
            else: subset_2.append(self.set[elt])

        self.partition = [subset_1, subset_2]
        self.split = np.sum(subset_1) == np.sum(subset_2)

class Clique(sb.SBModel):

    def __init__(self, connectivity_matrix: np.ndarray, clique_size: int) -> None:
        super().__init__()
        self.connectivity_matrix = connectivity_matrix
        self.vertices = connectivity_matrix.shape[0]
        self.clique_size = clique_size
        self.clique_found = False
        self.clique_index = None

    def __to_Ising__(self) -> sb.Ising:
        
        J = .25 * self.connectivity_matrix -.5 * 10 * (self.clique_size + 1) * np.ones((self.vertices, self.vertices))
        J -= np.diag(np.diag(J))
        h = - self.clique_size * 10 * (self.clique_size + 1) * np.ones((self.vertices, 1)) - .125 * J @ np.ones((self.vertices, 1)) 

        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        
        clique_vertices = (ising.ground_state == 1).reshape(-1,)

        if np.sum((1 + ising.ground_state) / 2) != self.clique_size:
            self.clique_found = False
            self.clique_index = None
        else:
            self.clique_index = clique_vertices
            print(self.connectivity_matrix[clique_vertices, :][:, clique_vertices] + np.eye(self.clique_size))
            self.clique_found = np.all(self.connectivity_matrix[clique_vertices, :][:, clique_vertices] + np.eye(self.clique_size) == 1)