Background
==========

Ising model
-----------

An Ising problem, given a null-diagonal square symmetrical matrix :math:`J` of size
:math:`N \times N` and a vector :math:`h` of size :math:`N`, consists in finding the
spin vector :math:`\mathbf{s} = (s_{1}, ... s_{N})` called the *ground state*,
(each :math:`s_{i}` being equal to either 1 or -1) such that the following value,
called *Ising energy*, is minimal:

.. math::

    - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij}s_{i}s_{j} + \sum_{i=1}^{N} h_{i}s_{i}

This problem is known to be NP-hard but is very useful since it can be used in many sectors
such as finance, transportation or chemistry or derived as other well-know optimization problems
(QUBO, MAXCUT, Knapsack problem, etc.).

The Simulated Bifurcation algorithm was originally introduced to solve Ising problems by simulating the adiabatic evolution of spins in a quantum Hamiltonian system, but can also be generalized to a wider range of optimization problems.

Multivariate order 2 polynomials
--------------------------------

In the most general terms possible, the Ising model can be reformulated as the minimization or maximization of multivariate order 2 polynomial (MO2P) with spin, binary or integer input values to express a wide
range of combinatorial optimization problems spanning from NP-hard and NP-complete problems (Karp, QUBO, TSP, ...) to Linear Programming.
Such a MO2P is mathematically expressed as:

.. math::

    \sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij}x_{i}x_{j} + \sum_{i=1}^{N} l_{i}x_{i} + c

where :math:`Q` is a square matrix, :math:`l` a vector and :math:`c` a constant.

This can also be seen as the sum of a quadratic form, a linear form and a constant term.

.. math::

    \mathbf{x}^T Q \mathbf{x} + l^T \mathbf{x} + c

Simulated Bifurcation algorithm
-------------------------------

The Simulated Bifurcation (SB) algorithm is a fast and highly parallelizable
state-of-the-art algorithm for combinatorial optimization inspired by quantum
physics and spins dynamics. It relies on Hamiltonian quantum mechanics to find
local minima of Ising problems. More specifically, it
solves the Ising problem, an NP-hard optimization problem which consists
in finding the ground state of an Ising model. By extension, any MO2P expressed
as previously can be either maximized or minimized for a given type of inputs
(spin, binary, integer) using the Simulated Bifurcation.

This package provides:

1. GPU compatible implementation of the simulated bifurcation (SB) algorithm, a quantum physics inspired combinatorial optimization approximation algorithm.
2. Implementation of several common combinatorial optimization problems.
3. A polynomial API for further customization.

Several common combinatorial optimization problems are reframed as Ising
problems in the `models` module, e.g.: QUBO, knapsack, Markowitz model...
Polynomials over vectors whose entries are in {0, 1} or whose entries are
fixed bit-width integers are also implemented, as well as an abstract
polynomial class `IsingPolynomialInterface` for further customization.

The docstring examples assume that `torch` (PyTorch) has been imported and
that simulated_bifurcation has been imported as `sb`:

  >>> import torch
  >>> import simulated_bifurcation as sb

Code snippets are indicated by three greater-than signs:

  >>> x = 42
  >>> x = x + 1

SB Algorithm versions
~~~~~~~~~~~~~~~~~~~~~

The original version of the SB algorithm [2] is not implemented since it is
less efficient than the more recent variants of the SB algorithm described
in [3] :

- **ballistic SB (bSB)** : Uses the position of the particles for the position-based update of the momentums ; usually faster but less accurate.
- **discrete SB (dSB)** : Uses the sign of the position of the particles for the position-based update of the momentums ; usually slower but more accurate.

On top of these two variants, an additional thermal fluctuation term
can be added in order to help escape local optima [4] (HbSB and HdSB). 

Parallelization (multi-agent search)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Simulated Bifurcation algorithm is highly parallelizable since it only relies on 
linear matrices equations. To take advantage of this property, this implementation
offers the possibility to perform a multi-agent search of the optimal solution by
evolving several spin vectors in parallel (each one being called an *agent*).

GPU computation
~~~~~~~~~~~~~~~

This parallelization of the algorithm can also be utilized by performing calculations on GPUs to speed
them up significantly. This packages harnesses the efficiency of PyTorch to provide a powerful GPU
computation system to run the SB algorithm.

Early stopping
~~~~~~~~~~~~~~

The Simulated Bifurcation algorithm stops after a certain number of iterations or when a pre-defined
computation timeout is reached. However, this implementation comes with the possibility to perform
early stopping and save computation time by defining convergence conditions. 

At regular intervals (this interval being called a sampling period), the agents (spin vectors) are
sampled and compared with their previous state by comparing their Ising energy. If the energy is the
same, the stability period of the agent is increased. If an agent's stability period exceeds a
convergence threshold, it is considered to have converged and its state is frozen. If all agents converge
before the maximum number of iterations has been reached, the algorithm then stops earlier.

The purpose of sampling the spins at regular intervals is to decorrelate them and make their stability more
informative about their convergence (because the evolution of the spins is *slow* it is expected that
most of the spins will not change from a time step to the following).

Notes
~~~~~
The SB algorithm is an approximation algorithm, which implies that the
returned values may not correspond to global optima. Therefore, if some
constraints are embedded as penalties in the polynomial, that is adding
terms that ensure that any global maximum satisfies the constraints, the
return values may violate these constraints.

The hyperparameters of the SB algorithm which correspond to physical
constants have been fine-tuned (Goto et al.) to give the best results most
of the time. Nevertheless, the relevance of specific hyperparameters may
vary depending on the properties of the instances. They can respectively be
modified and reset through the `set_env` and `reset_env` functions.

By denoting :math:`N` the dimension of the instance, :math:`A` the number of
agents and :math:`\Omega` the maximum number of steps, the time complexity of
the SB algorithm is :math:`O(\Omega \times A \times N^2)` and the space complexity
is :math:`O(A \times N + N^2)`.

For instances in low dimension (~100), running computations on GPU is
slower than running computations on CPU unless a large number of
agents (~2000) is used.

References
~~~~~~~~~~
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
~~~~~~~~
Minimize a polynomial over :math:`\{0, 1\} \times \{0, 1\}`

  >>> matrix = torch.tensor([[1, -2], [0, 3]], dtype=torch.float32)
  >>> vector = torch.tensor([3.5, 2.2], dtype=torch.float32)
  >>> constant = 3.1415
  >>> best_vector, best_value = sb.minimize(
  ...     matrix, vector, constant, "binary"
  ... )
  >>> best_vector
  tensor([0, 0])
  >>> best_value
  3.1415

Instantiate a polynomial over vectors whose entries are 3-bits integers
({0, 1, 2, ..., 6, 7})

  >>> poly = sb.build_model(matrix, vector, constant, "int3")

Maximize the polynomial over vectors whose entries are 3-bits integers

  >>> best_vector, best_value = poly.maximize()

Evaluate the polynomial at a single point

  >>> point = torch.tensor([0, 0], dtype=torch.float32)
  >>> poly(point)
  3.1415

Evaluate the polynomial at several points simultaneously

  >>> points = torch.tensor(
  ...     [[3, 5], [0, 0], [7, 1], [2, 6]],
  ...     dtype=torch.float32,
  ... )
  >>> poly(points)
  tensor([0, 3, 1, 2])

Create a QUBO instance and minimize it using a GPU to run the SB algorithm

  >>> qubo = sb.models.QUBO(matrix, device="cuda")
  >>> best_vector, best_value = qubo.minimize()
