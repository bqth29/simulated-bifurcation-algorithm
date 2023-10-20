"""
Simulated Bifurcation
=====================

Provides

1. GPU compatible implementation of the simulated bifurcation (SB) algorithm, a quantum physics inspired combinatorial optimization approximation algorithm.
2. Implementation of several common combinatorial optimization problems.
3. A polynomial API for further customization.

The simulated bifurcation (SB) algorithm is a randomized approximation
algorithm for combinatorial optimization problems. More specifically, it
solves the Ising problem, an NP-hard optimization problem which consists
in finding the ground state of an Ising model. It corresponds to the
minimization (or equivalently maximization) of a multivariate degree 2
polynomial over vectors whose entries are in {-1, 1}. Such polynomial is
the sum of a quadratic form and a linear form plus a constant term :
`ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
or `x.T Q x + l.T x + c` in matrix notation,
where `Q` is a square matrix, `l` a vector a `c` a constant.

Several common combinatorial optimization problems are reframed as Ising
problems in the `models` module, e.g.: QUBO, knapsack, Markowitz model...
Polynomials over vectors whose entries are in {0, 1} or whose entries are
fixed bit-width integers are also implemented, as well as an abstract
polynomial class `BaseMultivariateQuadraticPolynomial` for further customization.

The docstring examples assume that `torch` (PyTorch) has been imported and
that simulated_bifurcation has been imported as `sb`:

  >>> import torch
  >>> import simulated_bifurcation as sb

Code snippets are indicated by three greater-than signs:

  >>> x = 42
  >>> x = x + 1

.. deprecated:: 1.2.1
    `BinaryPolynomial` will be modified in simulated-bifurcation 1.3.0, it
    is replaced by `BinaryQuadraticPolynomial` in prevision of the addition
    of multivariate polynomials of an arbitrary degree.
    `IntegerPolynomial` will be modified in simulated-bifurcation 1.3.0, it
    is replaced by `IntegerQuadraticPolynomial` in prevision of the
    addition of multivariate polynomials of an arbitrary degree.
    `SpinPolynomial` will be modified in simulated-bifurcation 1.3.0, it is
    replaced by `SpinQuadraticPolynomial` in prevision of the addition of
    multivariate polynomials of an arbitrary degree.

Notes
-----
The SB algorithm is an approximation algorithm, which implies that the
returned values may not correspond to global optima. Therefore, if some
constraints are embedded as penalties in the polynomial, that is adding
terms that ensure that any global maximum satisfies the constraints, the
return values may violate these constraints.

The SB algorithm uses a randomized initialization, and this package is
implemented with a PyTorch backend. To ensure a consistent initialization
when running the same script multiple times, use `torch.manual_seed`.
However, results may not be reproducible between CPU and GPU executions,
even when using identical seeds. Furthermore, certain PyTorch operations
are not deterministic.
For more comprehensive details on reproducibility, refer to the PyTorch
documentation available at:
https://pytorch.org/docs/stable/notes/randomness.html.

The original version of the SB algorithm [2] is not implemented since it is
less efficient than the more recent variants of the SB algorithm described
in [3] :

- ballistic SB : Uses the position of the particles for the position-based update of the momentums ; usually faster but less accurate.
- discrete SB : Uses the sign of the position of the particles for the position-based update of the momentums ; usually slower but more accurate.

On top of these two variants, an additional thermal fluctuation term
can be added in order to help escape local optima [4]. Use this
additional term by setting `heated=True`.

The hyperparameters of the SB algorithm which correspond to physical
constants have been fine-tuned (Goto et al.) to give the best results most
of the time. Nevertheless, the relevance of specific hyperparameters may
vary depending on the properties of the instances. They can respectively be
modified and reset through the `set_env` and `reset_env` functions.

The time complexity is O(`max_steps` * `agents` * M^2) where M is the
dimension of the instance. The space complexity O(M^2 + `agents` * M).

For instances in low dimension (~100), running computations on GPU is
slower than running computations on CPU unless a large number of
agents (~2000) is used.

References
----------
[1] https://en.wikipedia.org/wiki/Ising_model

[2] Hayato Goto et al., "Combinatorial optimization by simulating adiabatic
bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5, eaav2372(2019).
DOI:10.1126/sciadv.aav2372

[3] Hayato Goto et al., "High-performance combinatorial optimization based
on classical mechanics". Sci. Adv.7, eabe7953(2021).
DOI:10.1126/sciadv.abe7953

[4] Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
fluctuation". Commun Phys 5, 153 (2022).
https://doi.org/10.1038/s42005-022-00929-9

Examples
--------
Minimize a polynomial over {0, 1} x {0, 1}
>>> matrix = torch.tensor([[1, -2], [0, 3]], dtype=torch.float32)
>>> vector = torch.tensor([3.5, 2.2], dtype=torch.float32)
>>> constant = 3.14
>>> best_vector, best_value = sb.minimize(
...     matrix, vector, constant, "binary"
... )
>>> best_vector
tensor([0., 0.])
>>> best_value
tensor(3.14)

Instantiate a polynomial over vectors whose entries are 3-bits integers
({0, 1, 2, ..., 6, 7})

  >>> poly = sb.build_model(matrix, vector, constant, "int3")

Maximize the polynomial over vectors whose entries are 3-bits integers
>>> best_vector, best_value = poly.maximize()
>>> best_vector
tensor([0., 7.])
>>> best_value
tensor(165.54)

Evaluate the polynomial at a single point
>>> point = torch.tensor([6, 3], dtype=torch.float32)
>>> poly(point)
tensor(57.74)

Evaluate the polynomial at several points simultaneously
>>> points = torch.tensor(
...     [[3, 5], [4, 4], [7, 1], [2, 6]],
...     dtype=torch.float32,
... )
>>> poly(points)
tensor([78.64, 57.94, 67.84, 111.34])

Create a QUBO instance and minimize it using a GPU to run the SB algorithm
>>> qubo = sb.models.QUBO(matrix, device="cuda")
>>> best_vector, best_value = qubo.minimize()  # Output is located on GPU
>>> best_vector
tensor([0., 0.], device='cuda:0')

"""


from . import models
from .optimizer import ConvergenceWarning, get_env, reset_env, set_env
from .polynomial import (
    BinaryPolynomial,
    BinaryQuadraticPolynomial,
    IntegerPolynomial,
    IntegerQuadraticPolynomial,
    SpinPolynomial,
    SpinQuadraticPolynomial,
)
from .simulated_bifurcation import build_model, maximize, minimize, optimize

reset_env()


# !MDC{set}{__version__ = "{version}"}
__version__ = "1.2.1.dev0"
