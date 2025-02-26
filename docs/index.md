# Simulated Bifurcation

```{toctree}
:hidden:
getting_started.md
background.md
```

```{toctree}
:caption: Package documentation
:hidden:
main_features.md
modules.md
```

<!-- TODO: add logo -->

The **Simulated Bifurcation** (SB) algorithm is a fast and highly parallelizable state-of-the-art algorithm for combinatorial optimization inspired by quantum physics and spins dynamics. It relies on Hamiltonian quantum mechanics to find local minima of **Ising** problems. The last accuracy tests showed a median optimality gap of less than 1% on high-dimensional instances.

This open-source package utilizes **PyTorch** to leverage GPU computations, harnessing the high potential for parallelization offered by the SB algorithm.

It also provides an API to define Ising models or other NP-hard and NP-complete problems (QUBO, Karp problems, ...) that can be solved using the SB algorithm.
