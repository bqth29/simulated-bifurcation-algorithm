from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Union, final

import numpy as np
import torch

from ..ising_core import IsingCore


class IsingPolynomialInterface(ABC):

    """
    Abstract class to implement an order two multivariate polynomial that can
    be translated as an equivalent Ising problem to be solved with the
    Simulated Bifurcation algorithm.

    The polynomial is the combination of a quadratic and a linear form plus a
    constant term:

    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`

    where `Q` is a square matrix, `l` a vector a `c` a constant.
    """

    def __init__(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        vector: Union[torch.Tensor, np.ndarray, None] = None,
        constant: Union[int, float, None] = None,
        accepted_values: Union[torch.Tensor, np.ndarray, list[int], None] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        matrix : Tensor | ndarray
            the square matrix that manages the order-two terms in the
            polynomial (quadratic form matrix).
        vector : Tensor | ndarray | None, optional
            the vector that manages the order-one terms in the polynomial
            (linear form vector). `None` means no vector (default is `None`)
        constant : float | int | None, optional
            the constant term of the polynomial. `None` means no constant term
            (default is `None`)
        accepted_values : Tensor | ndarray | List[int] | None, optional
            the values accepted as input values of the polynomial. Input with
            wrong values lead to a `ValueError` when evaluating the
            polynomial. `None` means no restriction in input values
            (default is `None`)
        dtype : torch.dtype, optional
            the dtype used to encode polynomial's coefficients (default is
            `float32`)
        device : str, optional
            the device on which to perform the computations of the Simulated
            Bifurcation algorithm (default `"cpu"`)
        """
        self.__check_device(device)
        self.__init_matrix(matrix, dtype, device)
        self.__init_vector(vector, dtype, device)
        self.__init_constant(constant, dtype, device)
        if accepted_values is not None:
            self.__accepted_values = torch.tensor(
                accepted_values, dtype=dtype, device=device
            )
        else:
            self.__accepted_values = None
        self.sb_result = None

    @property
    def matrix(self) -> torch.Tensor:
        return self.__matrix

    @property
    def vector(self) -> torch.Tensor:
        return self.__vector

    @property
    def constant(self) -> torch.Tensor:
        return self.__constant

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
        self,
        value: Union[torch.Tensor, np.ndarray],
        /,
        *,
        input_values_check: bool = True,
    ) -> Union[float, torch.Tensor]:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.tensor(value, dtype=self.dtype, device=self.device)
            except Exception as err:
                raise TypeError(f"Input value cannot be cast to Tensor.") from err
        if (
            input_values_check
            and self.__accepted_values is not None
            and torch.any(torch.isin(value, self.__accepted_values, invert=True))
        ):
            raise ValueError(
                f"Input values must all belong to {self.__accepted_values.tolist()}."
            )
        if value.shape == (self.dimension,):
            evaluation = (
                torch.vdot(value, torch.addmv(self.vector, self.matrix, value))
                + self.constant
            )
            return evaluation.item()
        if value.shape[-1] == self.dimension:
            quadratic_term = torch.nn.functional.bilinear(
                value,
                value,
                torch.unsqueeze(self.matrix, 0),
            )
            affine_term = value @ self.vector + self.constant
            evaluation = torch.squeeze(quadratic_term, -1) + affine_term
            return evaluation
        raise ValueError(
            f"Size of the input along the last axis should be "
            f"{self.dimension}, it is {value.shape[-1]}."
        )

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

    @staticmethod
    def __check_device(device: Union[str, torch.device]):
        if isinstance(device, torch.device):
            device = device.type
        elif not isinstance(device, str):
            raise TypeError(
                f"device should a string or a torch.device, received {device}"
            )
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, the SB algorithm cannot be run on GPU.\n"
                "See https://pytorch.org/get-started/locally/ for further information"
                "about installing with CUDA support."
            )  # pragma: no cover

    def __init_matrix(self, matrix: Iterable, dtype: torch.dtype, device: str) -> None:
        tensor_matrix = self.__cast_matrix_to_tensor(matrix, dtype, device)
        self.__check_square_matrix(tensor_matrix)
        self.__matrix = tensor_matrix
        self.__dimension = tensor_matrix.shape[0]

    def __init_vector(self, vector: Iterable, dtype: torch.dtype, device: str) -> None:
        tensor_vector = self.__cast_vector_to_tensor(vector, dtype, device)
        self.__check_vector_shape(tensor_vector)
        self.__vector = tensor_vector

    def __init_constant(
        self,
        constant: Union[float, int, None],
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> None:
        self.__constant = self.__cast_constant_to_float(constant, dtype, device)

    @staticmethod
    def __cast_matrix_to_tensor(
        matrix: Iterable, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        if isinstance(matrix, torch.Tensor):
            return matrix.to(dtype=dtype, device=device)
        try:
            return torch.tensor(matrix, dtype=dtype, device=device)
        except Exception as err:
            raise TypeError("Matrix cannot be cast to tensor.") from err

    def __cast_vector_to_tensor(
        self, vector: Optional[Iterable], dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        if vector is None:
            return torch.zeros(self.dimension, dtype=dtype, device=device)
        if isinstance(vector, torch.Tensor):
            return torch.squeeze(vector).to(dtype=dtype, device=device)
        try:
            return torch.squeeze(torch.tensor(vector, dtype=dtype, device=device))
        except Exception as err:
            raise TypeError("Vector cannot be cast to tensor.") from err

    @staticmethod
    def __cast_constant_to_float(
        constant: Union[float, int, None], dtype, device
    ) -> torch.Tensor:
        if constant is None:
            return torch.tensor(0.0, dtype=dtype, device=device)
        try:
            return torch.tensor(float(constant), dtype=dtype, device=device)
        except Exception as err:
            raise TypeError("Constant cannot be cast to float.") from err

    @staticmethod
    def __check_square_matrix(matrix: torch.Tensor) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"Matrix requires two dimension, got {matrix.ndim}.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")

    def __check_vector_shape(self, vector: torch.Tensor) -> None:
        allowed_shapes = [(self.dimension,), (self.dimension, 1), (1, self.dimension)]
        if vector.shape not in allowed_shapes:
            raise ValueError(
                f"Vector must be of size {self.dimension}, got {vector.shape}."
            )

    @abstractmethod
    def to_ising(self) -> IsingCore:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.

        Returns
        -------
        IsingCore
        """
        raise NotImplementedError

    @abstractmethod
    def convert_spins(self, ising: IsingCore) -> Optional[torch.Tensor]:
        """
        Retrieves information from the optimized equivalent Ising model.
        Returns the best found vector if `ising.ground_state` is not `None`.
        Returns `None` otherwise.

        Parameters
        ----------
        ising : IsingCore
            Equivalent Ising model of the problem.

        Returns
        -------
        Tensor
        """
        raise NotImplementedError

    @final
    def optimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        minimize: bool = True,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
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
        *
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
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        best_only : bool, optional
            if `True` only the best found solution to the optimization problem
            is returned, otherwise all the solutions found by the simulated
            bifurcation algorithm.
        minimize : bool, optional
            if `True` the optimization direction is minimization, otherwise it
            is maximization (default is True)

        Returns
        -------
        Tensor
        """
        if minimize:
            ising_equivalent = self.to_ising()
        else:
            ising_equivalent = -self.to_ising()
        ising_equivalent.optimize(
            agents,
            max_steps,
            ballistic,
            heated,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
        )
        self.sb_result = self.convert_spins(ising_equivalent)
        result = self.sb_result.t()
        evaluation = self(result)
        if best_only:
            i_min = torch.argmin(evaluation)
            result = result[i_min]
            evaluation = evaluation[i_min].item()
        return result, evaluation

    @final
    def minimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        Computes a local minimum of the model by optimizing
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
        *
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
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        best_only : bool, optional
            if `True` only the best found solution to the optimization problem
            is returned, otherwise all the solutions found by the simulated
            bifurcation algorithm.

        Returns
        -------
        Tensor
        """
        return self.optimize(
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            True,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
        )

    @final
    def maximize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        Computes a local maximum of the model by optimizing
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
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        best_only : bool, optional
            if `True` only the best found solution to the optimization problem
            is returned, otherwise all the solutions found by the simulated
            bifurcation algorithm.

        Returns
        -------
        Tensor
        """
        return self.optimize(
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            False,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
        )
