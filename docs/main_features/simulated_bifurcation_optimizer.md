# Simulated Bifurcation Optimizer

The package provides three functions to easily optimize quadratic polynomial
instances using the Simulated Bifurcation algorithm. This instance can be passed
as a SymPy polynomial expression or as a sequence of coefficient tensors. An optimization
domain shall also be specified.

## Parallelization (multi-agent search)

The Simulated Bifurcation algorithm is highly parallelizable since it only relies on 
linear matrices equations. To take advantage of this property, this implementation
offers the possibility to perform a multi-agent search of the optimal solution by
evolving several spin vectors in parallel (each one being called an *agent*).

## GPU computation

This parallelization of the algorithm can also be utilized by performing calculations on GPUs to speed
them up significantly. This packages harnesses the efficiency of PyTorch to provide a powerful GPU
computation system to run the SB algorithm.

## SB Algorithm versions

The original version of the SB algorithm[^1] is not implemented since it is
less efficient than the more recent variants of the SB algorithm[^2] :

- **ballistic SB (bSB)** : Uses the position of the particles for the position-based update of the momentums ; usually faster but less accurate.
- **discrete SB (dSB)** : Uses the sign of the position of the particles for the position-based update of the momentums ; usually slower but more accurate.

On top of these two variants, an additional thermal fluctuation term
can be added in order to help escape local optima[^3] (HbSB and HdSB). 

## Early stopping

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

## Parallelization (multi-agent search)

The Simulated Bifurcation algorithm is highly parallelizable since it only relies on 
linear matrices equations. To take advantage of this property, this implementation
offers the possibility to perform a multi-agent search of the optimal solution by
evolving several spin vectors in parallel (each one being called an *agent*).

## GPU computation

This parallelization of the algorithm can also be utilized by performing calculations on GPUs to speed
them up significantly. This packages harnesses the efficiency of PyTorch to provide a powerful GPU
computation system to run the SB algorithm.

## Available routines

The Simulated Bifurcation algorithm stops after a certain number of iterations or when a pre-defined
computation timeout is reached. However, this implementation comes with the possibility to perform
early stopping and save computation time by defining convergence conditions. 

At regular intervals, the state of the spins is sampled and compared with its previous value to calculate
their stability period. If an agent's stability period exceeds a convergence threshold, it is considered
to have converged and its value is frozen. If all agents converge before the maximum number of iterations
has been reached, the algorithm stops.

This functions also include a list of optional parameters to customize the call to SB by
setting the stopping strategy and the computation dtype and device.

```{eval-rst}
.. autofunction:: simulated_bifurcation.optimize
.. autofunction:: simulated_bifurcation.minimize
.. autofunction:: simulated_bifurcation.maximize
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
