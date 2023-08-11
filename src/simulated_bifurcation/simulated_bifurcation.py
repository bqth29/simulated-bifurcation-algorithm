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

"""


import re
from typing import Tuple, Union

import torch
from numpy import ndarray

from .polynomial import (
    BinaryPolynomial,
    IntegerPolynomial,
    IsingPolynomialInterface,
    SpinPolynomial,
)


def optimize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    input_type: str = "spin",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
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
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    r"""
    Optimize a multivariate degree 2 polynomial using the SB algorithm.

    The simulated bifurcated (SB) algorithm is a randomized approximation
    algorithm for combinatorial optimization problems.
    The optimization can either be a minimization or a maximization, and
    it is done over a discrete domain specified through `input_type`.
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
    input_type : {"spin", "binary", "int(\d+)"}, default=spin
        Domain over which the optimization is done.
        • "spin" : Optimize the polynomial over vectors whose entries are
        in {-1, 1}.
        • "binary" : Optimize the polynomial over vectors whose entries are
        in {0, 1}.
        • "int(\d+)" : Optimize the polynomial over vectors whose entries
        are n-bits non-negative integers, that is integers between 0 and
        2^n - 1 inclusive. "int(\d+)" represents any string starting with
        "int" and followed by a positive integer n, e.g. "int3", "int42".
    dtype : torch.dtype, default=torch.float32
        Data-type used for running the computations in the SB algorithm.
    device : str | torch.device, default="cpu"
        Device on which the SB algorithm is run. If available, use "cuda"
        to run the SB algorithm on GPU (much faster, especially for high
        dimensional instances or when running the algorithm with many
        agents).
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

    Returns
    -------
    result : ([`agents`], M) Tensor
        Best vector found, or all the vectors found is `best_only` is
        False.
    evaluation : float | (`agents`) Tensor
        Value of the polynomial at `result`.

    Other Parameters
    ----------------
    use_window : bool, default=True
        Whether to use the window as a stopping criterion.
    sampling_period : int, default=50
        Number of iterations between two consecutive spin samplings by the
        window.
    convergence_threshold : int, default=50
        Number of consecutive identical spin samplings considered as a
        proof of convergence by the window.
    Hyperparameters corresponding to physical constants :
        These parameters have been fine-tuned (Goto et al.) to give the
        best results most of the time. Nevertheless, the relevance of
        specific hyperparameters may vary depending on the properties of
        the instances. They can respectively be modified and reset through
        the `set_env` and `reset_env` functions.

    Raises
    ------
    ValueError
        If `input_type` is not one of {"spin", "binary", "int(\d+)"}.

    Warns
    -----
    If `use_window` is True and no agent has reached the convergence
    criterion defined by `sampling_period` and `convergence_threshold`
    within `max_steps` iterations, a warning is logged in the console.
    This is just an indication however; the returned solutions may still
    be of good quality. If the returned solutions are not of good quality,
    solutions include increasing `max_steps` (at the expense of runtime),
    changing the values of `ballistic` and `heated` to use different
    variants of the SB algorithm and changing the values of some
    hyperparameters corresponding to physical constants (advanced usage,
    see Other Parameters).

    Warnings
    --------
    The SB algorithm is an approximation algorithm, which implies that
    the returned values may not correspond to global optima. Therefore, if
    some constraints are embedded as penalties in the polynomial, that is
    adding terms that ensure that any global optimum satisfies the
    constraints, the return values may violate these constraints.

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
    dimension of the instance. The space complexity O(M^2 + `agents` * N).

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
    ...     Q, minimize=False, input_type="binary"
    ... )
    >>> best_vector
    tensor([0, 1])
    >>> best_value
    3

    Minimize Q and return all the solutions found using 42 agents
    >>> best_vectors, best_values = sb.optimize(
    ...     Q, input_type="binary", agents=42, best_only=False
    ... )
    >>> best_vectors.shape  # (agents, dimension of the instance)
    (42, 2)
    >>> best_values.shape  # (agents,)
    (42,)

    """
    model = build_model(
        matrix=matrix,
        vector=vector,
        constant=constant,
        input_type=input_type,
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
    )
    return result, evaluation


def minimize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    input_type: str = "spin",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
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
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    r"""
    Minimize a multivariate degree 2 polynomial using the SB algorithm.

    The simulated bifurcated (SB) algorithm is a randomized approximation
    algorithm for combinatorial optimization problems.
    The minimization is done over a discrete domain specified through
    `input_type`.
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
    input_type : {"spin", "binary", "int(\d+)"}, default=spin
        Domain over which the minimization is done.
        • "spin" : Minimize the polynomial over vectors whose entries are
        in {-1, 1}.
        • "binary" : Minimize the polynomial over vectors whose entries are
        in {0, 1}.
        • "int(\d+)" : Minimize the polynomial over vectors whose entries
        are n-bits non-negative integers, that is integers between 0 and
        2^n - 1 inclusive. "int(\d+)" represents any string starting with
        "int" and followed by a positive integer n, e.g. "int3", "int42".
    dtype : torch.dtype, default=torch.float32
        Data-type used for running the computations in the SB algorithm.
    device : str | torch.device, default="cpu"
        Device on which the SB algorithm is run. If available, use "cuda"
        to run the SB algorithm on GPU (much faster, especially for high
        dimensional instances or when running the algorithm with many
        agents).
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

    Returns
    -------
    result : ([`agents`], M) Tensor
        Best vector found, or all the vectors found is `best_only` is
        False.
    evaluation : float | (`agents`) Tensor
        Value of the polynomial at `result`.

    Other Parameters
    ----------------
    use_window : bool, default=True
        Whether to use the window as a stopping criterion.
    sampling_period : int, default=50
        Number of iterations between two consecutive spin samplings by the
        window.
    convergence_threshold : int, default=50
        Number of consecutive identical spin samplings considered as a
        proof of convergence by the window.
    Hyperparameters corresponding to physical constants :
        These parameters have been fine-tuned (Goto et al.) to give the
        best results most of the time. Nevertheless, the relevance of
        specific hyperparameters may vary depending on the properties of
        the instances. They can respectively be modified and reset through
        the `set_env` and `reset_env` functions.

    Raises
    ------
    ValueError
        If `input_type` is not one of {"spin", "binary", "int(\d+)"}.

    Warns
    -----
    If `use_window` is True and no agent has reached the convergence
    criterion defined by `sampling_period` and `convergence_threshold`
    within `max_steps` iterations, a warning is logged in the console.
    This is just an indication however; the returned solutions may still
    be of good quality. If the returned solutions are not of good quality,
    solutions include increasing `max_steps` (at the expense of runtime),
    changing the values of `ballistic` and `heated` to use different
    variants of the SB algorithm and changing the values of some
    hyperparameters corresponding to physical constants (advanced usage,
    see Other Parameters).

    Warnings
    --------
    The SB algorithm is an approximation algorithm, which implies that
    the returned values may not correspond to global minima. Therefore, if
    some constraints are embedded as penalties in the polynomial, that is
    adding terms that ensure that any global minimum satisfies the
    constraints, the return values may violate these constraints.

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
    dimension of the instance. The space complexity O(M^2 + `agents` * N).

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
    >>> best_vector, best_value = sb.minimize(Q, input_type="binary")
    >>> best_vector
    tensor([0, 0])
    >>> best_value
    0

    Return all the solutions found using 42 agents
    >>> best_vectors, best_values = sb.minimize(
    ...     Q, input_type="binary", agents=42, best_only=False
    ... )
    >>> best_vectors.shape  # (agents, dimension of the instance)
    (42, 2)
    >>> best_values.shape  # (agents,)
    (42,)

    """
    return optimize(
        matrix,
        vector,
        constant,
        input_type,
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
    )


def maximize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    input_type: str = "spin",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
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
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    r"""
    Maximize a multivariate degree 2 polynomial using the SB algorithm.

    The simulated bifurcated (SB) algorithm is a randomized approximation
    algorithm for combinatorial optimization problems.
    The maximization is done over a discrete domain specified through
    `input_type`.
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
    input_type : {"spin", "binary", "int(\d+)"}, default=spin
        Domain over which the maximization is done.
        • "spin" : Maximize the polynomial over vectors whose entries are
        in {-1, 1}.
        • "binary" : Maximize the polynomial over vectors whose entries are
        in {0, 1}.
        • "int(\d+)" : Maximize the polynomial over vectors whose entries
        are n-bits non-negative integers, that is integers between 0 and
        2^n - 1 inclusive. "int(\d+)" represents any string starting with
        "int" and followed by a positive integer n, e.g. "int3", "int42".
    dtype : torch.dtype, default=torch.float32
        Data-type used for running the computations in the SB algorithm.
    device : str | torch.device, default="cpu"
        Device on which the SB algorithm is run. If available, use "cuda"
        to run the SB algorithm on GPU (much faster, especially for high
        dimensional instances or when running the algorithm with many
        agents).
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

    Returns
    -------
    result : ([`agents`], M) Tensor
        Best vector found, or all the vectors found is `best_only` is
        False.
    evaluation : float | (`agents`) Tensor
        Value of the polynomial at `result`.

    Other Parameters
    ----------------
    use_window : bool, default=True
        Whether to use the window as a stopping criterion.
    sampling_period : int, default=50
        Number of iterations between two consecutive spin samplings by the
        window.
    convergence_threshold : int, default=50
        Number of consecutive identical spin samplings considered as a
        proof of convergence by the window.
    Hyperparameters corresponding to physical constants :
        These parameters have been fine-tuned (Goto et al.) to give the
        best results most of the time. Nevertheless, the relevance of
        specific hyperparameters may vary depending on the properties of
        the instances. They can respectively be modified and reset through
        the `set_env` and `reset_env` functions.

    Raises
    ------
    ValueError
        If `input_type` is not one of {"spin", "binary", "int(\d+)"}.

    Warns
    -----
    If `use_window` is True and no agent has reached the convergence
    criterion defined by `sampling_period` and `convergence_threshold`
    within `max_steps` iterations, a warning is logged in the console.
    This is just an indication however; the returned solutions may still
    be of good quality. If the returned solutions are not of good quality,
    solutions include increasing `max_steps` (at the expense of runtime),
    changing the values of `ballistic` and `heated` to use different
    variants of the SB algorithm and changing the values of some
    hyperparameters corresponding to physical constants (advanced usage,
    see Other Parameters).

    Warnings
    --------
    The SB algorithm is an approximation algorithm, which implies that
    the returned values may not correspond to global maxima. Therefore, if
    some constraints are embedded as penalties in the polynomial, that is
    adding terms that ensure that any global maximum satisfies the
    constraints, the return values may violate these constraints.

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
    dimension of the instance. The space complexity O(M^2 + `agents` * N).

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
    >>> best_vector, best_value = sb.maximize(Q, input_type="binary")
    >>> best_vector
    tensor([0, 1])
    >>> best_value
    3

    Return all the solutions found using 42 agents
    >>> best_vectors, best_values = sb.maximize(
    ...     Q, input_type="binary", agents=42, best_only=False
    ... )
    >>> best_vectors.shape  # (agents, dimension of the instance)
    (42, 2)
    >>> best_values.shape  # (agents,)
    (42,)

    """
    return optimize(
        matrix,
        vector,
        constant,
        input_type,
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
    )


def build_model(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray, None] = None,
    constant: Union[int, float, None] = None,
    input_type: str = "spin",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> IsingPolynomialInterface:
    r"""
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
    input_type : {"spin", "binary", "int(\d+)"}, default=spin
        Domain over which the maximization is done.
        - "spin" : Polynomial over vectors whose entries are in {-1, 1}.
        - "binary" : Polynomial over vectors whose entries are in {0, 1}.
        - "int(\d+)" : Polynomial over vectors whose entries are n-bits
        non-negative integers, that is integers between 0 and 2^n - 1
        inclusive. "int(\d+)" represents any string starting with "int" and
        followed by a positive integer n, e.g. "int3", "int42", ...
    dtype : torch.dtype, default=torch.float32
        Data-type used for storing the coefficients of the polynomial.
    device : str | torch.device, default="cpu"
        Device on which the polynomial is located. If available, use "cuda"
        to use the polynomial on a GPU.

    Returns
    -------
    SpinPolynomial | BinaryPolynomial | IntegerPolynomial
        The polynomial described by `matrix`, `vector` and `constant` on
        the domain specified by `input_type`.
        - `input_type="spin"` : SpinPolynomial.
        - `input_type="binary"` : BinaryPolynomial.
        - `input_type="binary"` : IntegerPolynomial.

    Raises
    ------
    ValueError
        If `input_type` is not one of {"spin", "binary", "int(\d+)"}.

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
        polynomial class `IsingPolynomialInterface`.
    models :
        Module containing the implementation of several common
        combinatorial optimization problems.

    Examples
    --------
    Instantiate a polynomial over {0, 1} x {0, 1}
    >>> Q = torch.tensor([[1, -2],
    ...                   [0, 3]])
    >>> poly = sb.build_model(Q, input_type="binary")

    Maximize the polynomial
    >>> best_vector, best_value = poly.maximize()
    >>> best_vector
    tensor([0, 1])
    >>> best_value
    3

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
    2

    Evaluate the polynomial at several points simultaneously
    >>> points = torch.tensor(
    ...     [[0, 0], [0, 1], [1, 0], [1, 1]],
    ...     dtype=torch.float32,
    ... )
    >>> poly(points)
    tensor([0, 3, 1, 2])

    """
    int_type_regex = re.compile(r"int(\d+)")
    if input_type == "spin":
        return SpinPolynomial(
            matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device
        )
    if input_type == "binary":
        return BinaryPolynomial(
            matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device
        )
    if int_type_regex.match(input_type):
        number_of_bits = int(int_type_regex.findall(input_type)[0])
        return IntegerPolynomial(
            matrix=matrix,
            vector=vector,
            constant=constant,
            dtype=dtype,
            device=device,
            number_of_bits=number_of_bits,
        )
    raise ValueError(r'Input type must match "spin", "binary" or "int(\d+)".')
