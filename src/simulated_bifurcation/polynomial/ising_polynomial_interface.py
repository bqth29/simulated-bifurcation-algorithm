from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Union, final

import numpy as np
import torch

from ..ising_core import IsingCore


class IsingPolynomialInterface(ABC):

    """
    Order two multivariate polynomial. Sum of a
    quadratic form, a linear form and a constant
    term.
    """

    def __init__(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        vector: Union[torch.Tensor, np.ndarray, None] = None,
        constant: Union[int, float, None] = None,
        accepted_values: Union[None, List[int]] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.__init_matrix(matrix, dtype, device)
        self.__init_vector(vector, dtype, device)
        self.__init_constant(constant)
        self.__accepted_values = (
            accepted_values[:] if accepted_values is not None else None
        )
        self.sb_result = None

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
    def __call__(
        self, value: Union[torch.Tensor, np.ndarray, List[List[float]]]
    ) -> Union[float, List[float]]:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.Tensor(value)
            except:
                raise TypeError(f"Input value cannot be cast to Tensor.")
        if (self.__accepted_values is not None) and (
            not np.all(np.isin(value.numpy(), self.__accepted_values))
        ):
            raise ValueError(
                f"Input values must all belong to {self.__accepted_values}."
            )
        if value.shape in [(self.dimension,), (self.dimension, 1), (1, self.dimension)]:
            value = value.reshape((-1, 1))
            value = (
                value.t() @ self.matrix @ value
                + value.t() @ self.vector
                + self.constant
            )
            return value.item()
        if value.shape[0] == self.dimension:
            values = (
                torch.einsum(
                    "ij, ji -> i", value.t(), self.matrix @ value + self.vector
                )
                + self.constant
            )
            return values.tolist()
        raise ValueError(f"Expected {self.dimension} rows, got {value.shape[0]}.")

    @final
    def __getitem__(self, coefficient: int) -> Union[torch.Tensor, float]:
        if coefficient == 0:
            return self.constant
        if coefficient == 1:
            return self.vector
        if coefficient == 2:
            return self.matrix
        raise ValueError("Only accepts 0, 1 or 2 as arguments.")

    @final
    def __len__(self) -> int:
        return self.__dimension

    def __init_matrix(self, matrix: Iterable, dtype: torch.dtype, device: str) -> None:
        tensor_matrix = self.__cast_matrix_to_tensor(matrix, dtype, device)
        self.__check_square_matrix(tensor_matrix)
        self.__matrix = tensor_matrix
        self.__dimension = tensor_matrix.shape[0]

    def __init_vector(self, vector: Iterable, dtype: torch.dtype, device: str) -> None:
        tensor_vector = self.__cast_vector_to_tensor(vector, dtype, device)
        self.__check_vector_shape(tensor_vector)
        self.__vector = tensor_vector

    def __init_constant(self, constant: Union[float, int, None]) -> None:
        self.__constant = self.__cast_constant_to_float(constant)

    @staticmethod
    def __cast_matrix_to_tensor(
        matrix: Iterable, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        try:
            return torch.Tensor(matrix).to(device=device, dtype=dtype)
        except:
            raise TypeError("Matrix cannot be cast to tensor.")

    @staticmethod
    def __check_square_matrix(matrix: torch.Tensor) -> None:
        if len(matrix.shape) != 2:
            raise ValueError(f"Matrix requires two dimension, got {len(matrix.shape)}.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")

    def __cast_vector_to_tensor(
        self, vector: Union[Iterable, None], dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        if vector is None:
            return torch.zeros(self.dimension, dtype=dtype, device=device)
        try:
            return torch.Tensor(vector).to(device=device, dtype=dtype)
        except:
            raise TypeError("Vector cannot be cast to tensor.")

    def __check_vector_shape(self, vector: torch.Tensor) -> None:
        allowed_shapes = [(self.dimension,), (self.dimension, 1), (1, self.dimension)]
        if vector.shape not in allowed_shapes:
            raise ValueError(
                f"Vector must be of size {self.dimension}, got {vector.shape}."
            )

    def __cast_constant_to_float(self, constant: Union[float, int, None]) -> float:
        if constant is None:
            return 0.0
        try:
            return float(constant)
        except:
            raise TypeError("Constant cannot be cast to float.")

    @abstractmethod
    def to_ising(self) -> IsingCore:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_spins(self, ising: IsingCore) -> Optional[torch.Tensor]:
        """
        Retrieves information from the optimized equivalent Ising model.
        Returns the best found vector if ising.ground_state is not None.
        Returns None otherwise.

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
        verbose: bool = True,
        minimize: bool = True,
    ) -> torch.Tensor:
        """
        Computes a local extremum of the model by optimizing
        the equivalent Ising model using the Simulated Bifurcation (SB)
        algorithm.

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually faster but less accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually slower but more accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However, a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps to explore the solution space
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
        ising_equivalent = self.to_ising() if minimize else -self.to_ising()
        ising_equivalent.optimize(
            convergence_threshold,
            sampling_period,
            max_steps,
            agents,
            use_window,
            ballistic,
            heat,
            verbose,
        )
        self.sb_result = self.convert_spins(ising_equivalent)
        return self.sb_result
