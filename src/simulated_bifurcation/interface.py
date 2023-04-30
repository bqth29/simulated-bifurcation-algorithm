from . import *
from abc import ABC, abstractmethod
from typing import List, Union, final


"""
Variants of the Ising model with binary or integer vectors.
"""

class IsingInterface(ABC):

    """
    An abstract class to adapt optimization problems as Ising problems.
    """

    def __init__(self, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> None:
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def __to_Ising__(self) -> Ising:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.
        Thus, there may be no scientific signification of this
        equivalent model.

        Returns
        -------
        Ising
        """
        raise NotImplementedError

    @abstractmethod
    def __from_Ising__(self, ising: Ising) -> None:
        """
        Retrieves information from the optimized equivalent Ising model.
        Modifies the object's attributes in place.

        Parameters
        ----------
        ising : Ising
            equivalent Ising model of the problem
        """
        raise NotImplementedError

    @final
    def optimize(
        self,
        time_step: float = .1,
        convergence_threshold: int = 50,
        sampling_period: int = 50,
        max_steps: int = 10000,
        agents: int = 128,
        pressure_slope: float = .01,
        gerschgorin: bool = False,
        use_window: bool = True,
        ballistic: bool = False,
        heat_parameter: float = None,
        verbose: bool = True
    ) -> None:
        """
        Computes an approximated solution of the Ising problem using the
        Simulated Bifurcation algorithm. The ground state in modified in place.
        It should correspond to a local minimum for the Ising energy function.

        The Simulated Bifurcation (SB) algorithm mimics Hamiltonian dynamics to
        make spins evolve throughout time. It uses a symplectic Euler scheme
        for this purpose.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually slower but more accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually faster but less accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the Euler scheme, a number of maximum steps
        needs to be specified. However a refined way to stop is also possible
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
        spin vector (energy-wise) is kept and used as the new Ising model'
        ground state.

        Parameters
        ----------

        - Euler scheme parameters

        time_step : float, optional
            step size for the time discretization (default is 0.01)
        symplectic_parameter : int | 'inf', optional
            symplectic parameter for the Euler's scheme (default is 2)
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 35)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 60000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 20)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)

        - Quantum parameters

        detuning_frequency : float, optional
            detuning frequency of the Hamiltonian (default is 1.0)
        pressure_slope : float, optional
            pumping pressure's linear slope allowing adiabatic evolution
            (default is 0.01)
        final_pressure : float | None, optional
            pumping pressure's maximum value; if None, no maximum value is set
            (default is None)
        xi0 : float | 'gerschgorin' | None, optional
            weighting coefficient in the Hamiltonian; if None it will be
            computed based on the J matrix (default is None)
        heat_parameter : float, optional
            heat parameter for the heated SB algorithm (default is 0.06)

        - Others

        verbose : bool, optional
            whether to display evolution information or not (default is True)

        See Also
        --------

        For more information on the Hamiltonian parameters, check
        `SymplecticEulerScheme`.

        Notes
        -----

        For low dimensions, see the `comprehensive_search` method function
        instead that will always find the true optimal ground state.
        """
        ising_equivalent = self.__to_Ising__()
        ising_equivalent.optimize(
            time_step,
            convergence_threshold,
            sampling_period,
            max_steps,
            agents,
            pressure_slope,
            gerschgorin,
            use_window,
            ballistic,
            heat_parameter,
            verbose
        )
        self.__from_Ising__(ising_equivalent)

class Binary(IsingInterface):

    """
    Variant of an Ising model where the states vectors are binary values instead of spins.
    Given a symmetric matrix `M`and a vector `v`, the value to minimize is 

    `-0.5 * ΣΣ M(i,j)b(i)b(j) + Σ v(i)b(i)`

    where the `b(i)`'s values are either `0` or `1`.
    """

    def __init__(self, matrix: torch.Tensor,
                vector: torch.Tensor,
                dtype: torch.dtype=torch.float32,
                device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.matrix = matrix.to(dtype=dtype, device=device)
        self.dimension = matrix.shape[0]
        if vector is None:
            self.vector = torch.zeros((self.dimension, 1), device=device)
        else:
            self.vector = vector.reshape(-1, 1).to(dtype=dtype, device=device)
        self.solution = None

    @property
    def objective_value(self) -> Union[float, None]: return self(self.solution)

    def __len__(self): return self.matrix.shape[0]

    def __call__(self, binary_vector: torch.Tensor) -> Union[None, float, List[float]]:

        if binary_vector is None: return None

        elif not isinstance(binary_vector, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(binary_vector)}.")

        elif torch.any(torch.abs(2 * binary_vector - 1) != 1):
            raise ValueError('Binary values must be either 0 or 1.')

        elif binary_vector.shape in [(self.dimension,), (self.dimension, 1)]:
            binary_vector = binary_vector.reshape((-1, 1))
            M, v = self.matrix, self.vector.reshape((-1, 1))
            value = -.5 * binary_vector.t() @ M @ binary_vector + binary_vector.t() @ v
            return value.item()

        elif binary_vector.shape[0] == self.dimension:
            M, v = self.matrix, self.vector.reshape((-1, 1))
            values = torch.einsum('ij, ji -> i', binary_vector.t(), -.5 * M @ binary_vector + v)
            return values.tolist()

        else:
            raise ValueError(f"Expected {self.dimension} rows, got {binary_vector.shape[0]}.")
        
    def min(self, binary_vectors: torch.Tensor) -> torch.Tensor:

        """
        Returns the binary vector with the lowest objective value.
        """

        values = self(binary_vectors)
        best_value = argmin(values)
        return binary_vectors[:, best_value]

    def __to_Ising__(self) -> Ising:
        
        J = self.matrix
        h = 2 * self.vector - self.matrix @ torch.ones((len(self), 1), device=self.device)

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None:
        self.solution = .5 * (ising.ground_state + 1)


class Integer(IsingInterface):

    """
    Variant of an Ising model where the states vectors are positive integer values instead of spins.
    All these integer values are contained in the interval `[0; 2**N-1]` where `N` is an integer called
    the `number_of_bits`.

    Given a symmetric matrix `M`and a vector `v`, the value to minimize is 

    `-0.5 * ΣΣ M(i,j)e(i)e(j) + Σ v(i)e(i)`

    where the `e(i)`'s values are integer values of the range `[0; 2**N-1]`.
    """

    def __init__(self, matrix: torch.Tensor,
                vector: torch.Tensor,
                number_of_bits: int,
                dtype: torch.dtype=torch.float32,
                device: str = 'cpu') -> None:
        super().__init__(dtype, device)
        self.matrix = matrix.to(dtype=dtype, device=device)
        self.dimension = matrix.shape[0]
        if vector is None:
            self.vector = torch.zeros((self.dimension, 1), device=device)
        else:
            self.vector = vector.reshape(-1, 1).to(dtype=dtype, device=device)
        self.number_of_bits = number_of_bits
        self.conversion_matrix = self.__conversion_matrix__()
        self.solution = None

    @property
    def objective_value(self) -> Union[float, None]: return self(self.solution)

    def __len__(self): return self.matrix.shape[0]

    def __call__(self, integer_vector: torch.Tensor) -> Union[None, float, List[float]]:

        if integer_vector is None: return None

        elif not isinstance(integer_vector, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(integer_vector)}.")

        elif torch.any(integer_vector != torch.round(integer_vector)):
            raise ValueError('Values must be integers.')
        
        elif torch.any(integer_vector > 2 ** self.number_of_bits - 1):
            raise ValueError(f'All values must be inferior to {2 ** self.number_of_bits - 1}.')

        elif integer_vector.shape in [(self.dimension,), (self.dimension, 1)]:
            integer_vector = integer_vector.reshape((-1, 1))
            M, v = self.matrix, self.vector.reshape((-1, 1))
            value = -.5 * integer_vector.t() @ M @ integer_vector + integer_vector.t() @ v
            return value.item()

        elif integer_vector.shape[0] == self.dimension:
            M, v = self.matrix, self.vector.reshape((-1, 1))
            values = torch.einsum('ij, ji -> i', integer_vector.t(), -.5 * M @ integer_vector + v)
            return values.tolist()

        else:
            raise ValueError(f"Expected {self.dimension} rows, got {integer_vector.shape[0]}.")
        
    def min(self, integer_vectors: torch.Tensor) -> torch.Tensor:

        """
        Returns the integer vector with the lowest objective value.
        """

        values = self(integer_vectors)
        best_value = argmin(values)
        return integer_vectors[:, best_value]
    
    def __conversion_matrix__(self) -> torch.Tensor:

        """
        Generates the integer-binary conversion matrix with the model's dimensions.
        Returns
        -------
        numpy.ndarray
        """  

        matrix = torch.zeros(
            (self.dimension * self.number_of_bits, self.dimension),
           device=self.device
        )

        for a in range(self.dimension):
            for b in range(self.number_of_bits):

                matrix[a*self.number_of_bits+b][a] = 2.0**b

        return matrix   

    def __to_Ising__(self) -> Ising:
        
        J = self.conversion_matrix @ self.matrix @ self.conversion_matrix.t()
        h = 2 * self.conversion_matrix @ self.vector \
            - self.conversion_matrix @ self.matrix @ self.conversion_matrix.t() \
            @ torch.ones((self.dimension * self.number_of_bits, 1), device=self.device)

        return Ising(J, h, self.dtype, self.device)

    def __from_Ising__(self, ising: Ising) -> None:
        self.solution = .5 * self.conversion_matrix.t() @ (ising.ground_state + 1)
