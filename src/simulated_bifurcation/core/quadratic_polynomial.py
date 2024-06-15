"""
Implementation of the QuadraticPolynomial class.

QuadraticPolynomial is a utility class to implement multivariate
quadratic polynomials from SymPy polynomial expressions or tensors.
They can automatically be casted to Ising model so they can be optimized
using the Simulated Bifurcation algorithm on a given domain. The available
domains are:

- spin optimization: variables are either -1 or +1
- binary optimization: variables are either 0 or 1
- n-bits integer optimization : variables are all integer values in the range
    0 to 2^n - 1 (inclusive)

See Also
--------
Ising:
    Interface to the Simulated Bifurcation algorithm used for optimizing
    user-defined polynomial.

"""

import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..polynomial import Polynomial, PolynomialLike
from .ising import Ising
from .variable import Variable

INTEGER_REGEX = re.compile("^int[1-9][0-9]*$")
DOMAIN_ERROR = ValueError(
    f'Input type must be one of "spin" or "binary", or be a string starting'
    f'with "int" and be followed by a positive integer.\n'
    f"More formally, it should match the following regular expression.\n"
    f"{INTEGER_REGEX}\n"
    f'Examples: "int7", "int42", ...'
)


class QuadraticPolynomialError(ValueError):
    def __init__(self, degree: int) -> None:
        super().__init__(f"Expected a degree 2 polynomial, got {degree}.")


class QuadraticPolynomial(Polynomial):
    """
    Internal implementation of a multivariate quadratic polynomial.

    A multivariate quadratic polynomial is the sum of a quadratic form and a
    linear form plus a constant term: `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`.
    In matrix notation, this gives: `x.T Q x + l.T x + c`, where `Q` is a
    square matrix, `l` a vector and `c` a constant.

    Multivariate quadratic polynomials are a common interface to express several
    optimization problems defined of different domains including:

    - spin optimization: variables are either -1 or +1
    - binary optimization: variables are either 0 or 1
    - n-bits integer optimization : variables are all integer values in the range
      0 to 2^n - 1 (inclusive)

    A multivariate quadratic polynomial defined on a given domain can be casted to
    an equivalent Ising model and thus optimized using the Simulated Bifurcation
    algorithm. The notion of equivalence means that finding the ground state of this
    Ising model is strictly equivalent to finding the vector solution that
    minimizes/maximizes the original polynomial when converted back to the original
    domain (for Ising optimization is in spins).

    Parameters
    ----------
    polynomial : PolynomialLike
        Source data of the multivariate quadratic polynomial to optimize. It can
        be a SymPy polynomial expression or tensors/arrays of coefficients.
        If tensors/arrays are provided, the monomial degree associated to
        the coefficients is the number of dimensions of the tensor/array,
        and all dimensions must be equal. The quadratic tensor must be square
        and symmetric and is mandatory. The linear tensor must be 1-dimensional
        and the constant term can either be a float/int or a 0-dimensional tensor.
        Both are optional. Tensors can be passed in an arbitrary order.

    Keyword-Only Parameters
    -----------------------
    dtype : torch.dtype, default=torch.float32
        Data-type used for the polynomial data.
    device : str | torch.device, default="cpu"
        Device on which the polynomial data is defined.

    Examples
    --------
    (Option 1) Instantiate a polynomial from tensors

      >>> Q = torch.tensor([[1, -2],
      ...                   [0, 3]])
      >>> poly = QuadraticPolynomial(Q)

    (Option 2) Instantiate a polynomial from a SymPy expression

      >>> x, y = sympy.symbols("x y")
      >>> expression = sympy.poly(x**2 - 2 * x * y + 3 * y**2)
      >>> poly = QuadraticPolynomial(expression)

    Maximize the polynomial over {0, 1} x {0, 1}

      >>> best_vector, best_value = poly.maximize(domain="binary")
      >>> best_vector
      tensor([0, 1])
      >>> best_value
      tensor(3)

    Return all the solutions found using 42 agents

      >>> best_vectors, best_values = poly.maximize(
      ...      agents=42, best_only=False
      ... )
      >>> best_vectors.shape  # (agents, dimension of the instance)
      (42, 2)
      >>> best_values.shape  # (agents,)
      (42,)

    Evaluate the polynomial at a single point

      >>> point = torch.tensor([1, 1], dtype=torch.float32)
      >>> poly(point)
      tensor(2)

    Evaluate the polynomial at several points simultaneously

      >>> points = torch.tensor(
      ...     [[0, 0], [0, 1], [1, 0], [1, 1]],
      ...     dtype=torch.float32,
      ... )
      >>> poly(points)
      tensor([0, 3, 1, 2])

    Migrate the polynomial to the GPU for faster computation

      >>> poly.to(device="cuda")

    Maximize this polynomial over {0, 1, ..., 14, 15} x {0, 1, ..., 14, 15}
    (outputs are located on the GPU)

      >>> best_vector, best_value = poly.maximize(domain="int4)
      >>> best_vector
      tensor([ 0., 15.], device='cuda:0')
      >>> best_value
      tensor(675., device='cuda:0')

    Evaluate this polynomial at a given point

      >>> point = torch.tensor([12, 7], dtype=torch.float32)
      >>> point = point.to(device="cuda")  # send tensor to GPU
      >>> poly(point)  # (output is located on GPU)
      tensor(123., device='cuda:0')

    """

    def __init__(
        self,
        *polynomial_like: PolynomialLike,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(*polynomial_like, dtype=dtype, device=device)
        self.sb_result = None
        if self.degree != 2:
            raise QuadraticPolynomialError(self.degree)

    def __call__(self, value: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.tensor(value, dtype=self.dtype, device=self.device)
            except Exception as err:
                raise TypeError("Input value cannot be cast to Tensor.") from err

        if value.shape[-1] != self.n_variables:
            raise ValueError(
                f"Size of the input along the last axis should be "
                f"{self.n_variables}, it is {value.shape[-1]}."
            )

        quadratic_term = torch.nn.functional.bilinear(
            value,
            value,
            torch.unsqueeze(self[2], 0),
        )
        affine_term = value @ self[1] + self[0]
        evaluation = torch.squeeze(quadratic_term, -1) + affine_term
        return evaluation

    def __get_variables(self, domain: Union[str, List[str]]) -> List[Variable]:
        if isinstance(domain, str):
            return [Variable.from_str(domain) for _ in range(self.n_variables)]
        if len(domain) != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} domains to be provided, got {len(domain)}."
            )
        return [Variable.from_str(variable_domain) for variable_domain in domain]

    def to_ising(self, domain: Union[str, List[str]]) -> Ising:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.

        Parameters
        ----------
        domain : str
            Domain over which the optimization is done.

            - "spin" : Optimize the polynomial over vectors whose entries are
              in {-1, 1}.
            - "binary" : Optimize the polynomial over vectors whose entries are
              in {0, 1}.
            - "int..." : Optimize the polynomial over vectors whose entries
              are n-bits non-negative integers, that is integers between 0 and
              2^n - 1 inclusive. "int..." represents any string starting with
              "int" and followed by a positive integer n, e.g. "int3", "int42".

            If the variables have different domains, a list of string with the
            same length as the number of variables can be provided instead.

        Returns
        -------
        Ising
            The equivalent Ising model to optimize with the Simulated
            Bifurcation algorithm.

        Raises
        ------
        ValueError
            If `domain` (or any domain in case a list is passed) is not one
            of {"spin", "binary", "int..."}, where "int..." designates any
            string starting with "int" and followed by a positive integer,
            or more formally, any string matching the regular expression
            `^int[1-9][0-9]*$`.
        ValueError
            If `domain` is used as a list of optimization domains with a
            length different from the number of variables.

        """
        variables = self.__get_variables(domain=domain)
        spin_identity_vector = QuadraticPolynomial.__spin_identity_vector(
            variables=variables, dtype=self.dtype, device=self.device
        )
        spin_weighted_integer_to_binary_matrix = (
            spin_identity_vector + 1
        ) * QuadraticPolynomial.__integer_to_binary_matrix(
            variables=variables, dtype=self.dtype, device=self.device
        )
        symmetric_quadratic_tensor = (self[2] + self[2].t()) / 2
        left_integer_to_binary_conversion = (
            spin_weighted_integer_to_binary_matrix.t() @ symmetric_quadratic_tensor
        )
        J = (
            -0.5
            * left_integer_to_binary_conversion
            @ spin_weighted_integer_to_binary_matrix
        )
        h = (
            0.5 * spin_weighted_integer_to_binary_matrix.t() @ self[1].reshape(-1, 1)
            - J.sum(axis=1).reshape(-1, 1)
            - left_integer_to_binary_conversion @ spin_identity_vector
        ).reshape(
            -1,
        )
        torch.diag(J)[...] = 0
        return Ising(J, h, self.dtype, self.device)

    def convert_spins(
        self, ising: Ising, domain: Union[str, List[str]]
    ) -> Optional[torch.Tensor]:
        """
        Retrieves information from the optimized equivalent Ising model.
        Returns the best found vector if `ising.ground_state` is not `None`.
        Returns `None` otherwise.

        Parameters
        ----------
        ising : IsingCore
            Equivalent Ising model to optimized with the Simulated
            Bifurcation algorithm.
        domain : str
            Domain over which the optimization is done.

            - "spin" : Optimize the polynomial over vectors whose entries are
              in {-1, 1}.
            - "binary" : Optimize the polynomial over vectors whose entries are
              in {0, 1}.
            - "int..." : Optimize the polynomial over vectors whose entries
              are n-bits non-negative integers, that is integers between 0 and
              2^n - 1 inclusive. "int..." represents any string starting with
              "int" and followed by a positive integer n, e.g. "int3", "int42".

            If the variables have different domains, a list of string with the
            same length as the number of variables can be provided instead.

        Returns
        -------
        Tensor

        Raises
        ------
        ValueError
            If `domain` (or any domain in case a list is passed) is not one
            of {"spin", "binary", "int..."}, where "int..." designates any
            string starting with "int" and followed by a positive integer,
            or more formally, any string matching the regular expression
            `^int[1-9][0-9]*$`.
        ValueError
            If `domain` is used as a list of optimization domains with a
            length different from the number of variables.
        """
        variables = self.__get_variables(domain=domain)
        spin_identity_vector = QuadraticPolynomial.__spin_identity_vector(
            variables=variables, dtype=self.dtype, device=self.device
        )
        spin_weighted_integer_to_binary_matrix = (
            spin_identity_vector + 1
        ) * QuadraticPolynomial.__integer_to_binary_matrix(
            variables=variables, dtype=self.dtype, device=self.device
        )
        return (
            None
            if ising.computed_spins is None
            else (
                0.5
                * spin_weighted_integer_to_binary_matrix
                @ (ising.computed_spins + 1)
                - spin_identity_vector
            )
        )

    def optimize(
        self,
        domain: Union[str, List[str]],
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
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.
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
            ising_equivalent = self.to_ising(domain)
        else:
            ising_equivalent = -self.to_ising(domain)
        ising_equivalent.minimize(
            agents,
            max_steps,
            ballistic,
            heated,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )
        self.sb_result = self.convert_spins(ising_equivalent, domain)
        result = self.sb_result.t().to(dtype=self.dtype)
        evaluation = self(result)
        if best_only:
            i_best = torch.argmin(evaluation) if minimize else torch.argmax(evaluation)
            result = result[i_best]
            evaluation = evaluation[i_best]
        return result, evaluation

    def minimize(
        self,
        domain: Union[str, List[str]],
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
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.
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
            domain,
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
            timeout=timeout,
        )

    def maximize(
        self,
        domain: Union[str, List[str]],
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
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.
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
            domain,
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
            timeout=timeout,
        )

    @staticmethod
    def __spin_identity_vector(
        variables: List[Variable], dtype: torch.dtype, device: Union[str, torch.device]
    ) -> torch.Tensor:
        vector = torch.zeros(len(variables), dtype=dtype, device=device)
        for ind, variable in enumerate(variables):
            vector[ind] = int(variable.is_spin)
        return vector.reshape(-1, 1)

    @staticmethod
    def __integer_to_binary_matrix(
        variables: List[Variable], dtype: torch.dtype, device: Union[str, torch.device]
    ) -> torch.Tensor:
        original_dimension = len(variables)
        new_dimension = np.sum([variable.encoding_bits for variable in variables])
        matrix = torch.zeros(
            original_dimension, new_dimension, dtype=dtype, device=device
        )
        column_offset = 0
        for row, variable in enumerate(variables):
            for col in range(variable.encoding_bits):
                matrix[row][column_offset + col] = 2**col
            column_offset += variable.encoding_bits
        return matrix
