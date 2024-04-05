# Simulated Bifurcation algorithm

The Simulated Bifurcation (SB) algorithm[^1][^2][^3] is a fast and highly parallelizable
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

```python
>>> import torch
>>> import simulated_bifurcation as sb
```

Code snippets are indicated by three greater-than signs:

```python
>>> x = 42
>>> x = x + 1
```

## Notes

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

## Examples

Minimize a polynomial over :math:`\{0, 1\} \times \{0, 1\}`

```python
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
```

Instantiate a polynomial over vectors whose entries are 3-bits integers
({0, 1, 2, ..., 6, 7})

```python
>>> poly = sb.build_model(matrix, vector, constant, "int3")
```

Maximize the polynomial over vectors whose entries are 3-bits integers

```python
>>> best_vector, best_value = poly.maximize()
```

Evaluate the polynomial at a single point

```python
>>> point = torch.tensor([0, 0], dtype=torch.float32)
>>> poly(point)
3.1415
```

Evaluate the polynomial at several points simultaneously

```python
>>> points = torch.tensor(
...     [[3, 5], [0, 0], [7, 1], [2, 6]],
...     dtype=torch.float32,
... )
>>> poly(points)
tensor([0, 3, 1, 2])
```

Create a QUBO instance and minimize it using a GPU to run the SB algorithm

```python
>>> qubo = sb.models.QUBO(matrix, device="cuda")
>>> best_vector, best_value = qubo.minimize()
```

[^1]: Hayato Goto et al., "Combinatorial optimization by simulating adiabatic
bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5, eaav2372(2019).
DOI:10.1126/sciadv.aav2372

[^2]: Hayato Goto et al., "High-performance combinatorial optimization based
on classical mechanics". Sci. Adv.7, eabe7953(2021).
DOI:10.1126/sciadv.abe7953

[^3]: Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
fluctuation". Commun Phys 5, 153 (2022).
https://doi.org/10.1038/s42005-022-00929-9
