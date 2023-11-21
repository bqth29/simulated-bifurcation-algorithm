"""
Module defining high-level routines for a basic usage of the
simulated_bifurcation package.

Available routines
------------------
optimize:
    Optimize a multivariate degree 2 polynomial using the SB algorithm.
minimize:
    Minimize a multivariate degree 2 polynomial using the SB algorithm.
maximize:
    Maximize a multivariate degree 2 polynomial using the SB algorithm.
build_model:
    Instantiate a multivariate degree 2 polynomial over a given domain.

See Also
--------
models:
    Package containing the implementation of several common combinatorial
    optimization problems.

"""


import re
import warnings
from typing import Optional, Tuple, Union

import torch
from numpy import ndarray

from .polynomial import (
    BaseMultivariateQuadraticPolynomial,
    BinaryQuadraticPolynomial,
    IntegerQuadraticPolynomial,
    SpinQuadraticPolynomial,
)


def optimize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    agents: int = 128,
    max_steps: int = 10_000,
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
    input_type: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize a multivariate degree 2 polynomial using the SB algorithm.

    The simulated bifurcated (SB) algorithm is a randomized approximation
    algorithm for combinatorial optimization problems.
    The optimization can either be a minimization or a maximization, and
    it is done over a discrete domain specified through `domain`.
    The polynomial is the sum of a quadratic form and a linear form plus
    a constant term:
    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
    or `x.T Q x + l.T x + c` in matrix notation,
    where `Q` is a square matrix, `l` a vector a `c` a constant.

    Parameters
    ----------
    matrix : (M, M) Tensor | ndarray
        Matrix corresponding to the quadratic terms of the polynomial
        (quadratic form). It should be a square matrix, but not necessarily
        symmetric.
    vector : (M,) Tensor | ndarray | None, optional
        Vector corresponding to the linear terms of the polynomial (linear
        form). The default is None which signifies there are no linear
        terms, that is `vector` is the null vector.
    constant : int | float | None, optional
        Constant of the polynomial. The default is None which signifies
        there is no constant term, that is `constant` = 0.
    domain : {"spin", "binary", "int..."}, default="spin"
        Domain over which the optimization is done.
        • "spin" : Optimize the polynomial over vectors whose entries are
        in {-1, 1}.
        • "binary" : Optimize the polynomial over vectors whose entries are
        in {0, 1}.
        • "int..." : Optimize the polynomial over vectors whose entries
        are n-bits non-negative integers, that is integers between 0 and
        2^n - 1 inclusive. "int..." represents any string starting with
        "int" and followed by a positive integer n, e.g. "int3", "int42".
    dtype : torch.dtype, default=torch.float32
        Data-type used for running the computations in the SB algorithm.
    device : str | torch.device, default="cpu"
        Device on which the SB algorithm is run. If available, use "cuda"
        to run the SB algorithm on GPU (much faster, especially for high
        dimensional instances or when running the algorithm with many
        agents). Output tensors are located on this device.
    agents : int, default=128
        Number of simultaneous execution of the SB algorithm. This is much
        faster than sequentially running the SB algorithm `agents` times.
    max_steps : int, default=10_000
        Number of iterations after which the algorithm is stopped
        regardless of whether convergence has been achieved.
    best_only : bool, default=True
        If True, return only the best vector found and the value of the
        polynomial at this vector. Otherwise, returns all the vectors
        found by the SB algorithm and the values of polynomial at these
        points.
    ballistic : bool, default=False
        Whether to use the ballistic or the discrete SB algorithm.
        See Notes for further information about the variants of the SB
        algorithm.
    heated : bool, default=False
        Whether to use the heated or non-heated SB algorithm.
        See Notes for further information about the variants of the SB
        algorithm.
    minimize : bool, default=True
        If True, minimizes the polynomial over the specified domain.
        Otherwise, the polynomial is maximized.
    verbose : bool, default=True
        Whether to display a progress bar to monitor the progress of the
        algorithm.
    input_type : deprecated, use `domain` instead.

    Returns
    -------
    result : ([`agents`], M) Tensor
        Best vector found, or all the vectors found is `best_only` is
        False.
    evaluation : ([`agents`],) Tensor
        Value of the polynomial at `result`.

    Other Parameters
    ----------------
    use_window : bool, default=True
        Whether to use the window as a stopping criterion: an agent is said
        to have converged if its energy has not changed over the last
        `convergence_threshold` energy samplings (done every
        `sampling_period` steps).
    sampling_period : int, default=50
        Number of iterations between two consecutive energy samplings by
        the window.
    convergence_threshold : int, default=50
        Number of consecutive identical energy samplings considered as a
        proof of convergence by the window.
    timeout : float | None, default=None
        Time, in seconds, after which the simulation will be stopped.
        None means no timeout.
    Hyperparameters corresponding to physical constants :
        These parameters have been fine-tuned (Goto et al.) to give the
        best results most of the time. Nevertheless, the relevance of
        specific hyperparameters may vary depending on the properties of
        the instances. They can respectively be modified and reset through
        the `set_env` and `reset_env` functions.

    Raises
    ------
    ValueError
        If `domain` is not one of {"spin", "binary", "int..."}, where
        "int..." designates any string starting with "int" and followed by
        a positive integer, or more formally, any string matching the
        following regular expression: ^int[1-9][0-9]*$.

    Warns
    -----
    If `use_window` is True and no agent has reached the convergence
    criterion defined by `sampling_period` and `convergence_threshold`
    within `max_steps` iterations, a warning is logged in the console.
    This is just an indication however; the returned vectors may still be
    of good quality. Solutions to this warning include:
        - increasing the time step in the SB algorithm (may decrease
            numerical stability), see the `set_env` function.
        - increasing `max_steps` (at the expense of runtime).
        - changing the values of `ballistic` and `heated` to use different
            variants of the SB algorithm.
        - changing the values of some hyperparameters corresponding to
            physical constants (advanced usage, see Other Parameters).

    Warnings
    --------
    Approximation algorithm:
        The SB algorithm is an approximation algorithm, which implies that
        the returned values may not correspond to global optima. Therefore,
        if some constraints are embedded as penalties in the polynomial,
        that is adding terms that ensure that any global optimum satisfies
        the constraints, the return values may violate these constraints.
    Non-deterministic behaviour:
        The SB algorithm uses a randomized initialization, and this package
        is implemented with a PyTorch backend. To ensure a consistent
        initialization when running the same script multiple times, use
        `torch.manual_seed`. However, results may not be reproducible
        between CPU and GPU executions, even when using identical seeds.
        Furthermore, certain PyTorch operations are not deterministic.
        For more comprehensive details on reproducibility, refer to the
        PyTorch documentation available at:
        https://pytorch.org/docs/stable/notes/randomness.html.

    See Also
    --------
    minimize : Alias for optimize(*args, **kwargs, minimize=True).
    maximize : Alias for optimize(*args, **kwargs, minimize=False).
    build_model : Create a polynomial object.
    models :
        Module containing the implementation of several common
        combinatorial optimization problems.

    Notes
    -----
    The original version of the SB algorithm [1] is not implemented since
    it is less efficient than the more recent variants of the SB algorithm
    described in [2]:
        ballistic SB : Uses the position of the particles for the
            position-based update of the momentums ; usually faster but
            less accurate. Use this variant by setting `ballistic=True`.
        discrete SB : Uses the sign of the position of the particles for
            the position-based update of the momentums ; usually slower
            but more accurate. Use this variant by setting
            `ballistic=False`.
    On top of these two variants, an additional thermal fluctuation term
    can be added in order to help escape local optima [3]. Use this
    additional term by setting `heated=True`.

    The time complexity is O(`max_steps` * `agents` * M^2) where M is the
    dimension of the instance. The space complexity O(M^2 + `agents` * M).

    For instances in low dimension (~100), running computations on GPU is
    slower than running computations on CPU unless a large number of
    agents (~2000) is used.

    References
    ----------
    [1] Hayato Goto et al., "Combinatorial optimization by simulating
    adiabatic bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5,
    eaav2372(2019). DOI:10.1126/sciadv.aav2372
    [2] Hayato Goto et al., "High-performance combinatorial optimization
    based on classical mechanics". Sci. Adv.7, eabe7953(2021).
    DOI:10.1126/sciadv.abe7953
    [3] Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
    fluctuation". Commun Phys 5, 153 (2022).
    https://doi.org/10.1038/s42005-022-00929-9

    Examples
    --------
    Maximize a polynomial over {0, 1} x {0, 1}
    >>> Q = torch.tensor([[1, -2],
    ...                   [0, 3]])
    >>> best_vector, best_value = sb.optimize(
    ...     Q, minimize=False, domain="binary"
    ... )
    >>> best_vector
    tensor([0., 1.])
    >>> best_value
    tensor(3.)

    Minimize a polynomial over {-1, 1} x {-1, 1} and return all the
    solutions found using 42 agents
    >>> best_vectors, best_values = sb.optimize(
    ...     Q, domain="spin", agents=42, best_only=False
    ... )
    >>> best_vectors.shape  # (agents, dimension of the instance)
    (42, 2)
    >>> best_values.shape  # (agents,)
    (42,)

    Minimize a polynomial over {0, 1, 2, ..., 6, 7} x {0, 1, 2, ..., 6, 7}
    using the GPU to run the SB algorithm. Outputs are located on the GPU.
    >>> best_vector, best_value = sb.optimize(
    ...     Q, domain="int3", device="cuda"
    ... )
    >>> best_vector
    tensor([0., 0.], device='cuda:0')
    >>> best_value
    tensor(0., device='cuda:0')

    """
    if input_type is not None:
        # 2023-11-21, 1.2.1
        warnings.warn(
            "`input_type` is deprecated as of simulated-bifurcation 1.2.1, and it will "
            "be removed in simulated-bifurcation 1.3.0. Please use `domain` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain = input_type

    model = build_model(
        matrix=matrix,
        vector=vector,
        constant=constant,
        domain=domain,
        dtype=dtype,
        device=device,
    )
    result, evaluation = model.optimize(
        agents=agents,
        max_steps=max_steps,
        best_only=best_only,
        ballistic=ballistic,
        heated=heated,
        minimize=minimize,
        verbose=verbose,
        use_window=use_window,
        sampling_period=sampling_period,
        convergence_threshold=convergence_threshold,
        timeout=timeout,
    )
    return result, evaluation


def minimize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    agents: int = 128,
    max_steps: int = 10_000,
    best_only: bool = True,
    ballistic: bool = False,
    heated: bool = False,
    verbose: bool = True,
    *,
    use_window: bool = True,
    sampling_period: int = 50,
    convergence_threshold: int = 50,
    timeout: Optional[float] = None,
    input_type: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimize a multivariate degree 2 polynomial using the SB algorithm.

    The simulated bifurcated (SB) algorithm is a randomized approximation
    algorithm for combinatorial optimization problems.
    The minimization is done over a discrete domain specified through
    `domain`.
    The polynomial is the sum of a quadratic form and a linear form plus
    a constant term:
    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
    or `x.T Q x + l.T x + c` in matrix notation,
    where `Q` is a square matrix, `l` a vector a `c` a constant.

    Parameters
    ----------
    matrix : (M, M) Tensor | ndarray
        Matrix corresponding to the quadratic terms of the polynomial
        (quadratic form). It should be a square matrix, but not necessarily
        symmetric.
    vector : (M,) Tensor | ndarray | None, optional
        Vector corresponding to the linear terms of the polynomial (linear
        form). The default is None which signifies there are no linear
        terms, that is `vector` is the null vector.
    constant : int | float | None, optional
        Constant of the polynomial. The default is None which signifies
        there is no constant term, that is `constant` = 0.
    domain : {"spin", "binary", "int..."}, default="spin"
        Domain over which the minimization is done.
        • "spin" : Minimize the polynomial over vectors whose entries are
        in {-1, 1}.
        • "binary" : Minimize the polynomial over vectors whose entries are
        in {0, 1}.
        • "int..." : Minimize the polynomial over vectors whose entries
        are n-bits non-negative integers, that is integers between 0 and
        2^n - 1 inclusive. "int..." represents any string starting with
        "int" and followed by a positive integer n, e.g. "int3", "int42".
    dtype : torch.dtype, default=torch.float32
        Data-type used for running the computations in the SB algorithm.
    device : str | torch.device, default="cpu"
        Device on which the SB algorithm is run. If available, use "cuda"
        to run the SB algorithm on GPU (much faster, especially for high
        dimensional instances or when running the algorithm with many
        agents). Output tensors are located on this device.
    agents : int, default=128
        Number of simultaneous execution of the SB algorithm. This is much
        faster than sequentially running the SB algorithm `agents` times.
    max_steps : int, default=10_000
        Number of iterations after which the algorithm is stopped
        regardless of whether convergence has been achieved.
    best_only : bool, default=True
        If True, return only the best vector found and the value of the
        polynomial at this vector. Otherwise, returns all the vectors
        found by the SB algorithm and the values of polynomial at these
        points.
    ballistic : bool, default=False
        Whether to use the ballistic or the discrete SB algorithm.
        See Notes for further information about the variants of the SB
        algorithm.
    heated : bool, default=False
        Whether to use the heated or non-heated SB algorithm.
        See Notes for further information about the variants of the SB
        algorithm.
    verbose : bool, default=True
        Whether to display a progress bar to monitor the progress of the
        algorithm.
    input_type : deprecated, use `domain` instead.

    Returns
    -------
    result : ([`agents`], M) Tensor
        Best vector found, or all the vectors found is `best_only` is
        False.
    evaluation : ([`agents`],) Tensor
        Value of the polynomial at `result`.

    Other Parameters
    ----------------
    use_window : bool, default=True
        Whether to use the window as a stopping criterion: an agent is said
        to have converged if its energy has not changed over the last
        `convergence_threshold` energy samplings (done every
        `sampling_period` steps).
    sampling_period : int, default=50
        Number of iterations between two consecutive energy samplings by
        the window.
    convergence_threshold : int, default=50
        Number of consecutive identical energy samplings considered as a
        proof of convergence by the window.
    timeout : float | None, default=None
        Time, in seconds, after which the simulation will be stopped.
        None means no timeout.
    Hyperparameters corresponding to physical constants :
        These parameters have been fine-tuned (Goto et al.) to give the
        best results most of the time. Nevertheless, the relevance of
        specific hyperparameters may vary depending on the properties of
        the instances. They can respectively be modified and reset through
        the `set_env` and `reset_env` functions.

    Raises
    ------
    ValueError
        If `domain` is not one of {"spin", "binary", "int..."}, where
        "int..." designates any string starting with "int" and followed by
        a positive integer, or more formally, any string matching the
        following regular expression: ^int[1-9][0-9]*$.

    Warns
    -----
    If `use_window` is True and no agent has reached the convergence
    criterion defined by `sampling_period` and `convergence_threshold`
    within `max_steps` iterations, a warning is logged in the console.
    This is just an indication however; the returned vectors may still be
    of good quality. Solutions to this warning include:
        - increasing the time step in the SB algorithm (may decrease
            numerical stability), see the `set_env` function.
        - increasing `max_steps` (at the expense of runtime).
        - changing the values of `ballistic` and `heated` to use different
            variants of the SB algorithm.
        - changing the values of some hyperparameters corresponding to
            physical constants (advanced usage, see Other Parameters).

    Warnings
    --------
    Approximation algorithm:
        The SB algorithm is an approximation algorithm, which implies that
        the returned values may not correspond to global minima. Therefore,
        if some constraints are embedded as penalties in the polynomial,
        that is adding terms that ensure that any global minimum satisfies
        the constraints, the return values may violate these constraints.
    Non-deterministic behaviour:
        The SB algorithm uses a randomized initialization, and this package
        is implemented with a PyTorch backend. To ensure a consistent
        initialization when running the same script multiple times, use
        `torch.manual_seed`. However, results may not be reproducible
        between CPU and GPU executions, even when using identical seeds.
        Furthermore, certain PyTorch operations are not deterministic.
        For more comprehensive details on reproducibility, refer to the
        PyTorch documentation available at:
        https://pytorch.org/docs/stable/notes/randomness.html.

    See Also
    --------
    maximize : Maximize a polynomial.
    build_model : Create a polynomial object.
    models :
        Module containing the implementation of several common
        combinatorial optimization problems.

    Notes
    -----
    The original version of the SB algorithm [1] is not implemented since
    it is less efficient than the more recent variants of the SB algorithm
    described in [2]:
        ballistic SB : Uses the position of the particles for the
            position-based update of the momentums ; usually faster but
            less accurate. Use this variant by setting `ballistic=True`.
        discrete SB : Uses the sign of the position of the particles for
            the position-based update of the momentums ; usually slower
            but more accurate. Use this variant by setting
            `ballistic=False`.
    On top of these two variants, an additional thermal fluctuation term
    can be added in order to help escape local minima [3]. Use this
    additional term by setting `heated=True`.

    The time complexity is O(`max_steps` * `agents` * M^2) where M is the
    dimension of the instance. The space complexity O(M^2 + `agents` * M).

    For instances in low dimension (~100), running computations on GPU is
    slower than running computations on CPU unless a large number of
    agents (~2000) is used.

    References
    ----------
    [1] Hayato Goto et al., "Combinatorial optimization by simulating
    adiabatic bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5,
    eaav2372(2019). DOI:10.1126/sciadv.aav2372
    [2] Hayato Goto et al., "High-performance combinatorial optimization
    based on classical mechanics". Sci. Adv.7, eabe7953(2021).
    DOI:10.1126/sciadv.abe7953
    [3] Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
    fluctuation". Commun Phys 5, 153 (2022).
    https://doi.org/10.1038/s42005-022-00929-9

    Examples
    --------
    Minimize a polynomial over {0, 1} x {0, 1}
    >>> Q = torch.tensor([[1, -2],
    ...                   [0, 3]])
    >>> best_vector, best_value = sb.minimize(Q, domain="binary")
    >>> best_vector
    tensor([0., 0.])
    >>> best_value
    tensor(0.)

    Return all the solutions found using 42 agents
    >>> best_vectors, best_values = sb.minimize(
    ...     Q, domain="binary", agents=42, best_only=False
    ... )
    >>> best_vectors.shape  # (agents, dimension of the instance)
    (42, 2)
    >>> best_values.shape  # (agents,)
    (42,)

    Minimize a polynomial over {0, 1, 2, ..., 6, 7} x {0, 1, 2, ..., 6, 7}
    using the GPU to run the SB algorithm. Outputs are located on the GPU.
    >>> best_vector, best_value = sb.minimize(
    ...     Q, domain="int3", device="cuda"
    ... )
    >>> best_vector
    tensor([0., 0.], device='cuda:0')
    >>> best_value
    tensor(0., device='cuda:0')

    """
    if input_type is not None:
        # 2023-11-21, 1.2.1
        warnings.warn(
            "`input_type` is deprecated as of simulated-bifurcation 1.2.1, and it will "
            "be removed in simulated-bifurcation 1.3.0. Please use `domain` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain = input_type

    return optimize(
        matrix,
        vector,
        constant,
        domain,
        dtype,
        device,
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
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    agents: int = 128,
    max_steps: int = 10_000,
    best_only: bool = True,
    ballistic: bool = False,
    heated: bool = False,
    verbose: bool = True,
    *,
    use_window: bool = True,
    sampling_period: int = 50,
    convergence_threshold: int = 50,
    timeout: Optional[float] = None,
    input_type: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Maximize a multivariate degree 2 polynomial using the SB algorithm.

    The simulated bifurcated (SB) algorithm is a randomized approximation
    algorithm for combinatorial optimization problems.
    The maximization is done over a discrete domain specified through
    `domain`.
    The polynomial is the sum of a quadratic form and a linear form plus
    a constant term:
    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
    or `x.T Q x + l.T x + c` in matrix notation,
    where `Q` is a square matrix, `l` a vector a `c` a constant.

    Parameters
    ----------
    matrix : (M, M) Tensor | ndarray
        Matrix corresponding to the quadratic terms of the polynomial
        (quadratic form). It should be a square matrix, but not necessarily
        symmetric.
    vector : (M,) Tensor | ndarray | None, optional
        Vector corresponding to the linear terms of the polynomial (linear
        form). The default is None which signifies there are no linear
        terms, that is `vector` is the null vector.
    constant : int | float | None, optional
        Constant of the polynomial. The default is None which signifies
        there is no constant term, that is `constant` = 0.
    domain : {"spin", "binary", "int..."}, default="spin"
        Domain over which the maximization is done.
        • "spin" : Maximize the polynomial over vectors whose entries are
        in {-1, 1}.
        • "binary" : Maximize the polynomial over vectors whose entries are
        in {0, 1}.
        • "int..." : Maximize the polynomial over vectors whose entries
        are n-bits non-negative integers, that is integers between 0 and
        2^n - 1 inclusive. "int..." represents any string starting with
        "int" and followed by a positive integer n, e.g. "int3", "int42".
    dtype : torch.dtype, default=torch.float32
        Data-type used for running the computations in the SB algorithm.
    device : str | torch.device, default="cpu"
        Device on which the SB algorithm is run. If available, use "cuda"
        to run the SB algorithm on GPU (much faster, especially for high
        dimensional instances or when running the algorithm with many
        agents). Output tensors are located on this device.
    agents : int, default=128
        Number of simultaneous execution of the SB algorithm. This is much
        faster than sequentially running the SB algorithm `agents` times.
    max_steps : int, default=10_000
        Number of iterations after which the algorithm is stopped
        regardless of whether convergence has been achieved.
    best_only : bool, default=True
        If True, return only the best vector found and the value of the
        polynomial at this vector. Otherwise, returns all the vectors
        found by the SB algorithm and the values of polynomial at these
        points.
    ballistic : bool, default=False
        Whether to use the ballistic or the discrete SB algorithm.
        See Notes for further information about the variants of the SB
        algorithm.
    heated : bool, default=False
        Whether to use the heated or non-heated SB algorithm.
        See Notes for further information about the variants of the SB
        algorithm.
    verbose : bool, default=True
        Whether to display a progress bar to monitor the progress of the
        algorithm.
    input_type : deprecated, use `domain` instead.

    Returns
    -------
    result : ([`agents`], M) Tensor
        Best vector found, or all the vectors found is `best_only` is
        False.
    evaluation : ([`agents`],) Tensor
        Value of the polynomial at `result`.

    Other Parameters
    ----------------
    use_window : bool, default=True
        Whether to use the window as a stopping criterion: an agent is said
        to have converged if its energy has not changed over the last
        `convergence_threshold` energy samplings (done every
        `sampling_period` steps).
    sampling_period : int, default=50
        Number of iterations between two consecutive energy samplings by
        the window.
    convergence_threshold : int, default=50
        Number of consecutive identical energy samplings considered as a
        proof of convergence by the window.
    timeout : float | None, default=None
        Time, in seconds, after which the simulation will be stopped.
        None means no timeout.
    Hyperparameters corresponding to physical constants :
        These parameters have been fine-tuned (Goto et al.) to give the
        best results most of the time. Nevertheless, the relevance of
        specific hyperparameters may vary depending on the properties of
        the instances. They can respectively be modified and reset through
        the `set_env` and `reset_env` functions.

    Raises
    ------
    ValueError
        If `domain` is not one of {"spin", "binary", "int..."}, where
        "int..." designates any string starting with "int" and followed by
        a positive integer, or more formally, any string matching the
        following regular expression: ^int[1-9][0-9]*$.

    Warns
    -----
    If `use_window` is True and no agent has reached the convergence
    criterion defined by `sampling_period` and `convergence_threshold`
    within `max_steps` iterations, a warning is logged in the console.
    This is just an indication however; the returned vectors may still be
    of good quality. Solutions to this warning include:
        - increasing the time step in the SB algorithm (may decrease
            numerical stability), see the `set_env` function.
        - increasing `max_steps` (at the expense of runtime).
        - changing the values of `ballistic` and `heated` to use different
            variants of the SB algorithm.
        - changing the values of some hyperparameters corresponding to
            physical constants (advanced usage, see Other Parameters).

    Warnings
    --------
    Approximation algorithm:
        The SB algorithm is an approximation algorithm, which implies that
        the returned values may not correspond to global maxima. Therefore,
        if some constraints are embedded as penalties in the polynomial,
        that is adding terms that ensure that any global maximum satisfies
        the constraints, the return values may violate these constraints.
    Non-deterministic behaviour:
        The SB algorithm uses a randomized initialization, and this package
        is implemented with a PyTorch backend. To ensure a consistent
        initialization when running the same script multiple times, use
        `torch.manual_seed`. However, results may not be reproducible
        between CPU and GPU executions, even when using identical seeds.
        Furthermore, certain PyTorch operations are not deterministic.
        For more comprehensive details on reproducibility, refer to the
        PyTorch documentation available at:
        https://pytorch.org/docs/stable/notes/randomness.html.

    See Also
    --------
    minimize : Minimize a polynomial.
    build_model : Create a polynomial object.
    models :
        Module containing the implementation of several common
        combinatorial optimization problems.

    Notes
    -----
    The original version of the SB algorithm [1] is not implemented since
    it is less efficient than the more recent variants of the SB algorithm
    described in [2]:
        ballistic SB : Uses the position of the particles for the
            position-based update of the momentums ; usually faster but
            less accurate. Use this variant by setting `ballistic=True`.
        discrete SB : Uses the sign of the position of the particles for
            the position-based update of the momentums ; usually slower
            but more accurate. Use this variant by setting
            `ballistic=False`.
    On top of these two variants, an additional thermal fluctuation term
    can be added in order to help escape local maxima [3]. Use this
    additional term by setting `heated=True`.

    The time complexity is O(`max_steps` * `agents` * M^2) where M is the
    dimension of the instance. The space complexity O(M^2 + `agents` * M).

    For instances in low dimension (~100), running computations on GPU is
    slower than running computations on CPU unless a large number of
    agents (~2000) is used.

    References
    ----------
    [1] Hayato Goto et al., "Combinatorial optimization by simulating
    adiabatic bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5,
    eaav2372(2019). DOI:10.1126/sciadv.aav2372
    [2] Hayato Goto et al., "High-performance combinatorial optimization
    based on classical mechanics". Sci. Adv.7, eabe7953(2021).
    DOI:10.1126/sciadv.abe7953
    [3] Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
    fluctuation". Commun Phys 5, 153 (2022).
    https://doi.org/10.1038/s42005-022-00929-9

    Examples
    --------
    Maximize a polynomial over {0, 1} x {0, 1}
    >>> Q = torch.tensor([[1, -2],
    ...                   [0, 3]])
    >>> best_vector, best_value = sb.maximize(Q, domain="binary")
    >>> best_vector
    tensor([0., 1.])
    >>> best_value
    tensor(3.)

    Return all the solutions found using 42 agents
    >>> best_vectors, best_values = sb.maximize(
    ...     Q, domain="binary", agents=42, best_only=False
    ... )
    >>> best_vectors.shape  # (agents, dimension of the instance)
    (42, 2)
    >>> best_values.shape  # (agents,)
    (42,)

    Maximize a polynomial over {0, 1, 2, ..., 6, 7} x {0, 1, 2, ..., 6, 7}
    using the GPU to run the SB algorithm. Outputs are located on the GPU.
    >>> best_vector, best_value = sb.maximize(
    ...     Q, domain="int3", device="cuda"
    ... )
    >>> best_vector
    tensor([0., 7.], device='cuda:0')
    >>> best_value
    tensor(147., device='cuda:0')

    """
    if input_type is not None:
        # 2023-11-21, 1.2.1
        warnings.warn(
            "`input_type` is deprecated as of simulated-bifurcation 1.2.1, and it will "
            "be removed in simulated-bifurcation 1.3.0. Please use `domain` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain = input_type

    return optimize(
        matrix,
        vector,
        constant,
        domain,
        dtype,
        device,
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


def build_model(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    input_type: Optional[str] = None,
) -> BaseMultivariateQuadraticPolynomial:
    """
    Instantiate a multivariate degree 2 polynomial over a given domain.

    The polynomial is the sum of a quadratic form and a linear form plus
    a constant term:
    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
    or `x.T Q x + l.T x + c` in matrix notation,
    where `Q` is a square matrix, `l` a vector a `c` a constant.

    Parameters
    ----------
    matrix : (M, M) Tensor | ndarray
        Matrix corresponding to the quadratic terms of the polynomial
        (quadratic form). It should be a square matrix, but not necessarily
        symmetric.
    vector : (M,) Tensor | ndarray | None, optional
        Vector corresponding to the linear terms of the polynomial (linear
        form). The default is None which signifies there are no linear
        terms, that is `vector` is the null vector.
    constant : int | float | None, optional
        Constant of the polynomial. The default is None which signifies
        there is no constant term, that is `constant` = 0.
    domain : {"spin", "binary", "int..."}, default="spin"
        Domain over which the maximization is done.
        - "spin" : Polynomial over vectors whose entries are in {-1, 1}.
        - "binary" : Polynomial over vectors whose entries are in {0, 1}.
        - "int..." : Polynomial over vectors whose entries are n-bits
        non-negative integers, that is integers between 0 and 2^n - 1
        inclusive. "int..." represents any string starting with "int" and
        followed by a positive integer n, e.g. "int3", "int42", ...
    dtype : torch.dtype, default=torch.float32
        Data-type used for storing the coefficients of the polynomial.
    device : str | torch.device, default="cpu"
        Device on which the polynomial is located. If available, use "cuda"
        to use the polynomial on a GPU.
    input_type : deprecated, use `domain` instead.

    Returns
    -------
    SpinQuadraticPolynomial | BinaryQuadraticPolynomial | IntegerQuadraticPolynomial
        The polynomial described by `matrix`, `vector` and `constant` on
        the domain specified by `domain`.
        - `domain="spin"` : SpinQuadraticPolynomial.
        - `domain="binary"` : BinaryQuadraticPolynomial.
        - `domain="int..."` : IntegerQuadraticPolynomial.

    Raises
    ------
    ValueError
        If `domain` is not one of {"spin", "binary", "int..."}, where
        "int..." designates any string starting with "int" and followed by
        a positive integer, or more formally, any string matching the
        following regular expression: ^int[1-9][0-9]*$.

    Warnings
    --------
    Calling a polynomial on a vector containing values which do not belong
    to the domain of the polynomial raises a ValueError, unless it is
    called while explicitly passing `input_values_check=False`.

    See Also
    --------
    minimize, maximize, optimize :
        Shorthands for polynomial creation and optimization.
    polynomial :
        Module providing some polynomial types as well as an abstract
        polynomial class `BaseMultivariateQuadraticPolynomial`.
    models :
        Module containing the implementation of several common
        combinatorial optimization problems.

    Examples
    --------
    Instantiate a polynomial over {0, 1} x {0, 1}
    >>> Q = torch.tensor([[1, -2],
    ...                   [0, 3]])
    >>> poly = sb.build_model(Q, domain="binary")

    Maximize the polynomial
    >>> best_vector, best_value = poly.maximize()
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

    Instantiate a polynomial over {0, 1, ..., 14, 15} x {0, 1, ..., 14, 15}
    and use it on the GPU
    >>> Q = torch.tensor([[1, -2],
    ...                   [0, 3]])
    >>> poly = sb.build_model(Q, domain="int4", device="cuda")

    Maximize this polynomial (outputs are located on the GPU)
    >>> best_vector, best_value = poly.maximize()
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
    if input_type is not None:
        # 2023-11-21, 1.2.1
        warnings.warn(
            "`input_type` is deprecated as of simulated-bifurcation 1.2.1, and it will "
            "be removed in simulated-bifurcation 1.3.0. Please use `domain` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        domain = input_type

    int_type_regex = "^int[1-9][0-9]*$"
    int_type_pattern = re.compile(int_type_regex)

    if domain == "spin":
        return SpinQuadraticPolynomial(
            matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device
        )
    if domain == "binary":
        return BinaryQuadraticPolynomial(
            matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device
        )
    if int_type_pattern.match(domain) is None:
        raise ValueError(
            f'Input type must be one of "spin" or "binary", or be a string starting'
            f'with "int" and be followed by a positive integer.\n'
            f"More formally, it should match the following regular expression.\n"
            f"{int_type_regex}\n"
            f'Examples: "int7", "int42", ...'
        )
    number_of_bits = int(domain[3:])
    return IntegerQuadraticPolynomial(
        matrix=matrix,
        vector=vector,
        constant=constant,
        dtype=dtype,
        device=device,
        number_of_bits=number_of_bits,
    )
