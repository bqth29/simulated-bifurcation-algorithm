from . import *
from numpy import sum


class NumberPartioning(SBModel):

    """
    A solver that separates a set of numbers into two subsets whose 
    respective sums are as close as possible.
    """

    def __init__(self, numbers: list, dtype: torch.dtype=torch.float32,
        device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.numbers = numbers
        self.partition = {
            'left': {'values': [], 'sum': None},
            'right': {'values': [], 'sum': None}
        }

    def __len__(self): return len(self.numbers)

    def __to_Ising__(self) -> Ising:
        
        tensor_numbers = torch.Tensor(self.numbers, device=self.device)
        J = -2 * tensor_numbers.reshape(-1, 1) @ tensor_numbers.reshape(1, -1)
        h = torch.zeros((len(self), 1))

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None:
        
        subset_left = []
        subset_right = []

        partition = ising.ground_state.reshape(-1,)

        for elt in range(len(self)):

            if partition[elt] > 0: subset_left.append(self.numbers[elt])
            else: subset_right.append(self.numbers[elt])

        self.partition['left']['values'] = subset_left
        self.partition['left']['sum'] = sum(subset_left)

        self.partition['right']['values'] = subset_right
        self.partition['right']['sum'] = sum(subset_right)

# class Clique(SBModel):

#     def __init__(self, adjacency_matrix: np.ndarray, clique_size: int) -> None:
#         super().__init__()
#         self.adjacency_matrix = adjacency_matrix
#         self.vertices = adjacency_matrix.shape[0]
#         self.clique_size = clique_size
#         self.clique_found = False
#         self.clique_index = None

#     def __to_Ising__(self) -> Ising:
        
#         J = .25 * self.adjacency_matrix -.5 * 10 * (self.clique_size + 1) * np.ones((self.vertices, self.vertices))
#         J -= np.diag(np.diag(J))
#         h = - self.clique_size * 10 * (self.clique_size + 1) * np.ones((self.vertices, 1)) - .125 * J @ np.ones((self.vertices, 1)) 

#         return Ising(J, h)

#     def __from_Ising__(self, ising: Ising) -> None:
        
#         clique_vertices = (ising.ground_state == 1).reshape(-1,)

#         if np.sum((1 + ising.ground_state) / 2) != self.clique_size:
#             self.clique_found = False
#             self.clique_index = None
#         else:
#             self.clique_index = clique_vertices
#             print(self.adjacency_matrix[clique_vertices, :][:, clique_vertices] + np.eye(self.clique_size))
#             self.clique_found = np.all(self.adjacency_matrix[clique_vertices, :][:, clique_vertices] + np.eye(self.clique_size) == 1)