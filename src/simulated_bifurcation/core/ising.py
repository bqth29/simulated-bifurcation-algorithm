"""
Implementation of the Ising class.

Ising is an interface to the Simulated Bifurcation algorithm and is
used for optimizing user-defined polynomial. See models.Ising for
an implementation of the Ising model which behaves like other models
and polynomials.

See Also
--------
models.Ising:
    Implementation of the Ising model which behaves like other models and
    polynomials.
QuadraticPolynomial:
    Class to implement multivariate quadratic polynomials from SymPy
    polynomial expressions or tensors that can be casted to Ising model
    for Simulated Bifurcation algorithm compatibility purposes.

"""

from typing import Literal, Optional, TypeVar, Union

import torch
from numpy import ndarray

from ..optimizer import SimulatedBifurcationEngine, SimulatedBifurcationOptimizer

# Workaround because `Self` type is only available in Python >= 3.11
SelfIsing = TypeVar("SelfIsing", bound="Ising")


class Ising(object):
    """
    Internal implementation of the Ising model.

    Solving an Ising problem means finding a spin vector `s` (with values
    in {-1, 1}) such that, given a matrix `J` with zero diagonal and a
    vector `h`, the following quantity - called Ising energy - is minimal
    (`s` is then called a ground state):
    `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`
    or `-0.5 x.T J x + h.T x` in matrix notation.

    Parameters
    ----------
    J: (M, M) Tensor
        Square matrix representing the quadratic part of the Ising model
        whose size is `M` the dimension of the problem.
    h: (M,) Tensor | None, optional
        Vector representing the linear part of the Ising model whose size
        is `M` the dimension of the problem. If this argument is not
        provided (`h is None`), it defaults to the null vector.
    dtype: torch.dtype, default=torch.float32
        Data-type used for storing the coefficients of the Ising model.
    device: str | torch.device, default="cpu"
        Device on which the instance is located.

    Attributes
    ----------
    dtype
    device
    dimension : int
        Size of the Ising problem, i.e. number of spins.
    computed_spins : (A, M) Tensor | None
        Spin vectors obtained by minimizing the Ising energy. None if no
        solving method has been called.
    J: (M, M) Tensor
        Square matrix representing the quadratic part of the Ising model
        whose size is `M` the dimension of the problem.
    h: (M,) Tensor
        Vector representing the linear part of the Ising model whose size
        is `M` the dimension of the problem.
    linear_term: bool
        Whether the model has a non-zero linear term.

    See Also
    --------
    models.Ising:
        An implementation of the Ising model which behaves like other
        models and polynomials.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Ising_model

    """

    def __init__(
        self,
        J: Union[torch.Tensor, ndarray],
        h: Union[torch.Tensor, ndarray, None] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._dtype = torch.float32 if dtype is None else dtype
        self._device = (
            torch.get_default_device() if device is None else torch.device(device)
        )

        if self._dtype not in [torch.float32, torch.float64]:
            raise ValueError(
                f"Simulated Bifurcation optimization can only be carried out with torch.float32 or torch.float64 dtypes, but got {dtype}."
            )

        if isinstance(J, ndarray):
            J = torch.from_numpy(J)
        if isinstance(h, ndarray):
            h = torch.from_numpy(h)

        if J.ndim != 2:
            raise ValueError(
                f"Expected J to be 2-dimensional, but got {J.ndim} dimensions."
            )
        rows, cols = J.shape
        if rows != cols:
            raise ValueError(
                f"Expected J to be square, but got {rows} rows and {cols} columns."
            )

        self._J = J.to(dtype=self._dtype, device=self._device)
        self._dimension = rows

        if h is None:
            self._h = torch.zeros(
                self._dimension, dtype=self._dtype, device=self._device
            )
        elif h.shape != (self._dimension,):
            raise ValueError(
                f"Expected the shape of h to be {self._dimension}, but got {tuple(h.shape)}."
            )
        else:
            self._h = h.to(dtype=self._dtype, device=self._device)

        self._has_linear_term = not torch.equal(
            self._h,
            torch.zeros(self._dimension, dtype=self._dtype, device=self._device),
        )

    def __neg__(self) -> SelfIsing:
        return self.__class__(-self._J, -self._h, self._dtype, self._device)

    def as_simulated_bifurcation_tensor(self) -> torch.Tensor:
        """
        Turn the instance into a tensor compatible with the SB algorithm.

        The SB algorithm runs on Ising models with no linear term, and
        whose matrix is symmetric and has only zeros on its diagonal.

        `self.J` is symmetrized and its diagonal get filled by zeros.

        If the Ising model has linear terms, `self.J` and `self.h` are
        gathered into a single tensor.

        The output matrix describes an equivalent Ising model in dimension
        `self.dimension + 1` with no linear term.

        Returns
        -------
        sb_tensor : Tensor
            Equivalent tensor compatible with the SB algorithm.

        Notes
        -----
        The output matrix is defined as the following block matrix.
        ```
            (            |    )
            (     J*     | -h )
            (____________|____)
            (    -h.T    |  0 )
        ```

        where `J* = (self.J + self.J.T) / 2.0` with a null diagonal.

        This matrix describes another Ising model `other` with no linear
        term in dimension `self.dimension + 1`, with the same minimal
        energy, and with a one to two correspondence between the ground
        states of the two models defined as follows.

        Ground states of `self` → Ground states of `other` ~ R^n x R

        s ↦ {(s, 1), (-s, -1)}

        """
        symmetrical_J = (self._J + self._J.t()) / 2.0  # symmetrize J
        torch.diagonal(symmetrical_J)[...] = 0.0  # remove diagonal
        if self._has_linear_term:
            sb_tensor = torch.zeros(
                (self._dimension + 1, self._dimension + 1),
                dtype=self._dtype,
                device=self._device,
            )
            sb_tensor[: self._dimension, : self._dimension] = symmetrical_J
            sb_tensor[: self._dimension, self._dimension] = -self._h
            sb_tensor[self._dimension, : self._dimension] = -self._h
            return sb_tensor
        else:
            return symmetrical_J

    def minimize(
        self,
        *,
        agents: int = 128,
        max_steps: int = 10000,
        mode: Literal["ballistic", "discrete"] = "ballistic",
        heated: bool = False,
        verbose: bool = True,
        early_stopping: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Minimize the energy of the Ising model using the Simulated Bifurcation
        algorithm.

        Parameters
        ----------
        agents : int, default=128
            Number of simultaneous execution of the SB algorithm. This is
            much faster than sequentially running the SB algorithm `agents`
            times.
        max_steps : int, default=10_000
            Number of iterations after which the algorithm is stopped
            regardless of whether convergence has been achieved.
        mode : "ballistic" | "discrete", optional, default = "ballistic"
            Whether to use the ballistic or the discrete SB algorithm.
            See Notes for further information about the variants of the SB
            algorithm.
        heated : bool, default=False
            Whether to use the heated or non-heated SB algorithm.
            See Notes for further information about the variants of the SB
            algorithm.
        verbose : bool, default=True
            Whether to display a progress bar to monitor the progress of
            the algorithm.
        early_stopping : bool, default=True, keyword-only
            Whether to use early stopping or not, making agents' convergence a
            stopping criterion. An agent is said to have converged if its energy
            has not changed over the last `convergence_threshold` energy samplings
            (done every `sampling_period` steps).
        sampling_period : int, default=50, keyword-only
            Number of iterations between two consecutive spins samplings used for
            early stopping.
        convergence_threshold : int, default=50, keyword-only
            Number of consecutive identical energy samplings considered as a
            proof of convergence of an agent.
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.

        Returns
        -------
        torch.Tensor
            The spins of all agents returned by the SB algorithm.

        Other Parameters
        ----------------
        Hyperparameters corresponding to physical constants :
            These parameters have been fine-tuned (Goto et al.) to give the
            best results most of the time. Nevertheless, the relevance of
            specific hyperparameters may vary depending on the properties
            of the instances. They can respectively be modified and reset
            through the `set_env` and `reset_env` functions.

        Warns
        -----
        If `early_stopping` is True and no agent has reached the convergence
        criterion defined by `sampling_period` and `convergence_threshold`
        within `max_steps` iterations, a warning is logged in the console.
        This is just an indication however; the returned vectors may still
        be of good quality. Solutions to this warning include:

        - increasing the time step in the SB algorithm (may decrease
          numerical stability), see the `set_env` function.
        - increasing `max_steps` (at the expense of runtime).
        - changing the values of `ballistic` and `heated` to use
          different variants of the SB algorithm.
        - changing the values of some hyperparameters corresponding to
          physical constants (advanced usage, see Other Parameters).

        Warnings
        --------
        Approximation algorithm:
            The SB algorithm is an approximation algorithm, which implies
            that the returned values may not correspond to global optima.
            Therefore, if some constraints are embedded as penalties in the
            polynomial, that is adding terms that ensure that any global
            optimum satisfies the constraints, the return values may
            violate these constraints.
        Non-deterministic behaviour:
            The SB algorithm uses a randomized initialization, and this
            package is implemented with a PyTorch backend. To ensure a
            consistent initialization when running the same script multiple
            times, use `torch.manual_seed`. However, results may not be
            reproducible between CPU and GPU executions, even when using
            identical seeds. Furthermore, certain PyTorch operations are
            not deterministic. For more comprehensive details on
            reproducibility, refer to the PyTorch documentation available
            at https://pytorch.org/docs/stable/notes/randomness.html.

        See Also
        --------
        models.Ising:
            Implementation of the Ising model which behaves like other
            models and polynomials.
        QuadraticPolynomial:
            Class to implement multivariate quadratic polynomials from SymPy
            polynomial expressions or tensors that can be casted to Ising model
            for Simulated Bifurcation algorithm compatibility purposes.

        Notes
        -----
        The original version of the SB algorithm [1] is not implemented
        since it is less efficient than the more recent variants of the SB
        algorithm described in [2]:

        - ballistic SB : Uses the position of the particles for the
          position-based update of the momentums ; usually faster but
          less accurate. Use this variant by setting
          `ballistic=True`.
        - discrete SB : Uses the sign of the position of the particles
          for the position-based update of the momentums ; usually
          slower but more accurate. Use this variant by setting
          `ballistic=False`.

        On top of these two variants, an additional thermal fluctuation
        term can be added in order to help escape local optima [3]. Use
        this additional term by setting `heated=True`.

        The space complexity O(M^2 + `agents` * M). The time complexity is
        O(`max_steps` * `agents` * M^2) where M is the dimension of the
        instance.

        For instances in low dimension (~100), running computations on GPU
        is slower than running computations on CPU unless a large number of
        agents (~2000) is used.

        References
        ----------
        [1] Hayato Goto et al., "Combinatorial optimization by simulating
        adiabatic bifurcations in nonlinear Hamiltonian systems". Sci.
        Adv.5, eaav2372(2019). DOI:10.1126/sciadv.aav2372
        [2] Hayato Goto et al., "High-performance combinatorial
        optimization based on classical mechanics". Sci. Adv.7,
        eabe7953(2021). DOI:10.1126/sciadv.abe7953
        [3] Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
        fluctuation". Commun Phys 5, 153 (2022).
        https://doi.org/10.1038/s42005-022-00929-9

        """
        engine = SimulatedBifurcationEngine.get_engine(mode)
        optimizer = SimulatedBifurcationOptimizer(
            agents,
            max_steps,
            timeout,
            engine,
            heated,
            verbose,
            sampling_period,
            convergence_threshold,
        )
        tensor = self.as_simulated_bifurcation_tensor()
        spins = optimizer.run_integrator(tensor, early_stopping)
        if self._has_linear_term:
            return spins[-1] * spins[:-1]
        else:
            return spins
