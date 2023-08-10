"""
Simulated Bifurcation
=====================

Provides
  1. GPU compatible implementation of the simulated bifurcation (SB)
  algorithm, a physics inspired combinatorial approximation algorithm.
  2. Implementation of several common combinatorial optimization problems.
  3. A polynomial API for further customization.

The simulated bifurcated (SB) algorithm is a randomized approximation
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
polynomial class `IsingPolynomialInterface` for further customization.

Notes
-----
The SB algorithm is an approximation algorithm, which implies that the
returned values may not correspond to global optima. Therefore, if some
constraints are embedded as penalties in the polynomial, that is adding
terms that ensure that any global maximum satisfies the constraints, the
return values may violate these constraints.

The original version of the SB algorithm [2] is not implemented since it is
less efficient than the more recent variants of the SB algorithm described
in [3] :
  ballistic SB : Uses the position of the particles for the position-based
    update of the momentums ; usually faster but less accurate.
  discrete SB : Uses the sign of the position of the particles for the
    position-based update of the momentums ; usually slower but more
    accurate.
On top of these two variants, an additional thermal fluctuation term
can be added in order to help escape local optima [4]. Use this
additional term by setting `heated=True`.

The hyperparameters of the SB algorithm which correspond to physical
constants have been fine-tuned  (Goto et al.) to give the best results most
of the time. Nevertheless, the relevance of specific  hyperparameters may
vary depending on the properties of the instances. They can respectively be
modified and reset through the `set_env` and `reset_env` functions.

The time complexity is O(`max_steps` * `agents` * M^2) where M is the
dimension of the instance. The space complexity O(M^2 + `agents` * N).

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
TODO

"""


from .optimizer import get_env, reset_env, set_env
from .polynomial import BinaryPolynomial, IntegerPolynomial, SpinPolynomial
from .simulated_bifurcation import build_model, maximize, minimize, optimize

reset_env()
