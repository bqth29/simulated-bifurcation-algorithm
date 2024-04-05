# Quadratic optimization

## Ising model

An Ising problem, given a null-diagonal square symmetrical matrix $J$ of size
$N \times N$ and a vector $h$ of size $N$, consists in finding the
spin vector $\mathbf{s} = (s_{1}, ... s_{N})$ called the *ground state*,
(each $s_{i}$ being equal to either 1 or -1) such that the following value,
called *Ising energy*, is minimal:

$$- \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij}s_{i}s_{j} + \sum_{i=1}^{N} h_{i}s_{i}$$

This problem is known to be NP-hard but is very useful since it can be used in many sectors
such as finance, transportation or chemistry or derived as other well-know optimization problems
(QUBO, MAXCUT, Knapsack problem, etc.).

The Simulated Bifurcation algorithm was originally introduced to solve Ising problems by simulating the adiabatic evolution of spins in a quantum Hamiltonian system, but can also be generalized to a wider range of optimization problems.

## Multivariate order 2 polynomials

In the most general terms possible, the Ising model can be reformulated as the minimization or maximization of multivariate order 2 polynomial (MO2P) with spin, binary or integer input values to express a wide
range of combinatorial optimization problems spanning from NP-hard and NP-complete problems (Karp, QUBO, TSP, ...) to Linear Programming.
Such a MO2P is mathematically expressed as:

$$\sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij}x_{i}x_{j} + \sum_{i=1}^{N} l_{i}x_{i} + c$$

where $Q$ is a square matrix, $l$ a vector and $c$ a constant.

This can also be seen as the sum of a quadratic form, a linear form and a constant term.

$$\mathbf{x}^T Q \mathbf{x} + l^T \mathbf{x} + c$$
