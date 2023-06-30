from .ising import Ising
from abc import ABC, abstractmethod
from typing import final, List, Union
import torch
import numpy as np


class IsingInterface(ABC):

    """
    Abstract class to derive optimization problems from Ising models.
    """

    def __init__(self, matrix: torch.Tensor, vector: torch.Tensor,
                 constant: Union[int, float], accepted_values: List[int],
                 dtype: torch.dtype, device: str) -> None:
        self.__matrix = matrix.to(dtype=dtype, device=device)
        self.__vector = vector.to(dtype=dtype, device=device)
        self.__constant = constant
        self.__accepted_values = accepted_values[:]
        self.__dimension = matrix.shape[0]

    #TODO: add import from numpy and check dimesions + None

    @property
    def matrix(self) -> torch.Tensor:
        return self.__matrix
    
    @property
    def vector(self) -> torch.Tensor:
        return self.__vector.reshape(-1, 1)
    
    @property
    def constant(self) -> float:
        return float(self.__constant)
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    
    @property
    def dtype(self) -> torch.dtype:
        return self.__matrix.dtype
    
    @property
    def device(self) -> torch.device:
        return self.__matrix.device

    @final
    def __call__(self, value: Union[torch.Tensor, np.ndarray, List[List[float]]]) -> Union[float, List[float]]:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.Tensor(value)
            except:
                raise TypeError(f"Input value cannot be cast to Tensor.")
        if not np.all(np.isin(value.numpy(), self.__accepted_values)):
            raise ValueError(f'Input values must all belong to {self.__accepted_values}.')
        if value.shape in [(self.dimension,), (self.dimension, 1), (1, self.dimension)]:
            value = value.reshape((-1, 1))
            value = value.t() @ self.matrix @ value + value.t() @ self.vector + self.constant
            return value.item()
        if value.shape[0] == self.dimension:
            values = torch.einsum('ij, ji -> i', value.t(), self.matrix @ value + self.vector) + self.constant
            return values.tolist()
        raise ValueError(f"Expected {self.dimension} rows, got {value.shape[0]}.")
    
    @final
    def __len__(self) -> int:
        return self.__dimension

    @abstractmethod
    def __to_ising(self) -> Ising:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.
        """
        raise NotImplementedError

    @abstractmethod
    def __from_ising(self, ising: Ising) -> torch.Tensor:
        """
        Retrieves information from the optimized equivalent Ising model.
        Returns the best found vector.

        Parameters
        ----------
        ising : Ising
            equivalent Ising model of the problem
        """
        raise NotImplementedError

    @final
    def optimize(
        self,
        convergence_threshold: int = 50,
        sampling_period: int = 50,
        max_steps: int = 10000,
        agents: int = 128,
        use_window: bool = True,
        ballistic: bool = False,
        heat: bool = False,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Computes a local extremum of the model by optimizing
        the equivalent Ising model using the Simulated
        Simulated Bifurcation (SB) algorithm. 

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually slower but more accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually faster but less accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps exploring the solution space
        and increases the probability of finding a better solution, though it
        also slightly increases the computation time. In the end, only the best
        spin vector (energy-wise) is kept and used as the new Ising model's
        ground state.

        Parameters
        ----------
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 50)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 10000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 128)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heat : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        """
        ising_equivalent = self.__to_ising()
        ising_equivalent.optimize(
            convergence_threshold, sampling_period,
            max_steps, agents, use_window, ballistic, heat, verbose
        )
        return self.__from_ising(ising_equivalent)
    
class SpinPolynomial(IsingInterface):

    def __init__(self, matrix: torch.Tensor, vector: Union[torch.Tensor, None] = None, constant: Union[float, int, None] = None,
                dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(matrix, vector, constant, [-1, 1], dtype, device)

    def to_ising(self) -> Ising:
        return Ising(self.matrix, self.vector, self.dtype, self.device)

    def from_ising(self, ising: Ising) -> torch.Tensor:
        return ising.ground_state


class BinaryPolynomial(IsingInterface):

    """
    Given a matrix `Q` (quadratic form), a vector `l`
    (linear form) and a constant `c`, the value to minimize is 
    `ΣΣ Q(i,j)b(i)b(j) + Σ l(i)b(i) + c` where the `b(i)`'s values
    are either `0` or `1`.
    """

    def __init__(self, matrix: torch.Tensor, vector: Union[torch.Tensor, None] = None, constant: Union[float, int, None] = None,
                dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(matrix, vector, constant, [0, 1], dtype, device)

    def to_ising(self) -> Ising:
        symmetrical_matrix = .5 * (self.matrix + self.matrix.t())
        J = -.5 * symmetrical_matrix
        h = .5 * self.vector + .5 * symmetrical_matrix @ torch.ones((len(self), 1), device=self.device)
        return Ising(J, h, self.dtype, self.device)

    def from_ising(self, ising: Ising) -> torch.Tensor:
        if ising.ground_state is not None:
            return .5 * (ising.ground_state + 1)
        

class IntegerPolynomial(IsingInterface):

    """
    Given a matrix `Q` (quadratic form), a vector `l`
    (linear form) and a constant `c`, the value to minimize is 
    `ΣΣ Q(i,j)n(i)n(j) + Σ l(i)n(i) + c` where the `n(i)`'s values
    are integers.
    """

    def __init__(self, matrix: torch.Tensor, vector: Union[torch.Tensor, None] = None, constant: Union[float, int, None] = None,
                number_of_bits: int = 1, dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(matrix, vector, constant, [*range(2**number_of_bits)], dtype, device)
        self.number_of_bits = number_of_bits
        self.__int_to_bin_matrix = IntegerPolynomial.integer_to_binary_matrix(self.dimension, self.number_of_bits, self.device)

    @staticmethod
    def integer_to_binary_matrix(dimension: int, number_of_bits: int, device: str) -> torch.Tensor:
        matrix = torch.zeros((dimension * number_of_bits, dimension), device=device)
        for row in range(dimension):
            for col in range(number_of_bits):
                matrix[row * number_of_bits + col][row] = 2.0**col
        return matrix  
    
    def to_ising(self) -> Ising:
        symmetrical_matrix = .5 * (self.matrix + self.matrix.t())
        J = -.5 * self.__int_to_bin_matrix @ symmetrical_matrix @ self.__int_to_bin_matrix.t()
        h = .5 * self.__int_to_bin_matrix @ self.vector \
            + .5 * self.__int_to_bin_matrix @ self.matrix @ self.__int_to_bin_matrix.t() \
            @ torch.ones((self.dimension * self.number_of_bits, 1), device=self.device)
        return Ising(J, h, self.dtype, self.device)

    def from_ising(self, ising: Ising) -> None:
        if ising.ground_state is not None:
            return .5 * self.__int_to_bin_matrix.t() @ (ising.ground_state + 1)

class QUBO(BinaryPolynomial):

    """
    Quadratic Unconstrained Binary Optimization

    Given a matrix `Q` the value to minimize is the quadratic form
    `ΣΣ Q(i,j)b(i)b(j)` where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(self, Q: torch.Tensor, dtype: torch.dtype=torch.float32, device: str = 'cpu') -> None:
        super().__init__(Q, None, None, dtype, device)
        self.Q = self.matrix
