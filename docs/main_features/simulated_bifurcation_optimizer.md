# Simulated Bifurcation Optimizer

This package provides an implementation of the Simulated Bifurcation algorithm enhanced by features...

It provides three optimization functions (`optimize`, `maximize` and `minimize`) that all share the same pool of parameters

> For the parameters relative to the quantum physics theory, see [Advanced Usages](advanced_usages.md).

These parameters are gathered and the following table and their usage and specific role in the optimizer are described thoughout this page.

> - The mandatory parameters are written in **bold**
> - Except for `polynomial` which is positional and must be defined as the first parameter, all other parameters are keyword-only and their order does not matter

| Parameter                                       | Type                    | Default value | Usage                                                                                                                                                                                 |
| ----------------------------------------------- | ----------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**`polynomial`**](#model-definition)           | `PolynomialLike`        |               | Quadratic model to optimize.                                                                                                                                                          |
| [`agents`](#parallelization-multi-agent-search) | `int`                   | `128`         | Model                                                                                                                                                                                 |
| [`ballistic`](#sb-algorithm-versions)           | `bool`                  | `True`        | Whether to use the ballistic version of the SB algorithm (bSB) or the discrete version (dSB).                                                                                         |
| [`best_only`](#outputs)                         | `bool`                  | `True`        | Whether to only return the best agent and its associated objective function value, or all agents at once.                                                                             |
| [`convergence_threshold`](#early-stopping)      | `int`                   | `50`          | Model                                                                                                                                                                                 |
| [`device`](#gpu-computation)                    | `str` or `torch.device` | `None`        | Device on which to run the optimization (CPU or GPU).                                                                                                                                 |
| [**`domain`**](#model-definition)               | `str`                   |               | Domain on which the optimization is carried out (spin, binary or integer values).                                                                                                     |
| [`dtype`](#model-definition)                    | `torch.dtype`           | `None`        | Model                                                                                                                                                                                 |
| [`heated`](#sb-algorithm-versions)              | `bool`                  | `False`       | Whether to use the heated version of the SB algorithm or not.                                                                                                                         |
| [`max_steps`](#number-of-iterations)            | `int`                   | `10000`       | Maximum number of iterations of the optimizer (one iteration is one step in the symplectic Eule scheme). If reached, the computation is stopped and the current results are returned. |
| [`sampling_period`](#early-stopping)            | `int`                   | `50`          | Model                                                                                                                                                                                 |
| [`timeout`](#computation-timeout)               | `int`                   | `None`        | Maximum computation time of the optimizer in seconds. If reached, the computation is stopped and the current results are returned.                                                    |
| [`use_window`](#early-stopping)                 | `bool`                  | `True`        | Model                                                                                                                                                                                 |
| [`verbose`](#display-the-evolution-status)      | `bool`                  | `True`        | Whether to display the evolution status of the optimizer with progress bars or not.                                                                                                   |

## Model definition

### Quadratic model

### Optimization domain

## Parallelization (multi-agent search)

The Simulated Bifurcation algorithm is highly parallelizable since it only relies on  linear matrices equations. To take advantage of this property, this implementation offers the possibility to perform a multi-agent search of the optimal solution by evolving several spin vectors in parallel (each one being called an *agent*).

Each agent is randomly initialized with values between -1 and 1 and evolves independently from the others. Using several agents provides an optimized wider exploration of the solution space time-wise.

> The number of agents is set using the `agents` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", agents=100)
> ```

## GPU computation

The parallelization of the algorithm can also be utilized by performing calculations on GPUs to speed them up significantly. This packages harnesses the efficiency of PyTorch to provide a powerful GPU computation system to run the SB algorithm.

> The computation device is set using the `device` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", device="cpu") # For CPU computation
> sb.minimize(polynomial, domain="spin", device="cuda") # For GPU computation
> ```

## SB Algorithm versions

The original version of the SB algorithm[^1] is not implemented since it is less efficient than the more recent variants of the SB algorithm[^2] :

- **ballistic SB (bSB)** : Uses the position of the particles for the position-based update of the momentums ; usually faster but less accurate.
- **discrete SB (dSB)** : Uses the sign of the position of the particles for the position-based update of the momentums ; usually slower but more accurate.

On top of these two variants, an additional thermal fluctuation term can be added in order to help escape local optima[^3] (HbSB and HdSB).

> The Simulated Bifurcation algorithm version is set using the `ballistic` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", ballistic=True) # Ballistic SB
> sb.minimize(polynomial, domain="spin", ballistic=False) # Discrete SB
> ```

## Stopping criteria

### Number of iterations

As in the original papers, the Simulated Bifurcation algorithm stops after a certain number of iterations which can be configured by the user.

> The maximum number of iterations is set using the `max_steps` parameter. If set to `None`, it will be ignored:
>
> ```python
> sb.minimize(polynomial, domain="spin", max_steps=10000) # Stop the computation after 10000 iterations
> ```

### Computation timeout

The implementation of the Simulated Bifurcation algorithm also offers the possibilty to interrupt the computation when a user-defined timeout is reached.

> The computation timeout (in seconds) is set using the `timeout` parameter. If set to `None`, it will be ignored:
>
> ```python
> sb.minimize(polynomial, domain="spin", timeout=60) # Stop the computation after 60 seconds
> ```

### Early stopping

Finally, this implementation comes with the possibility to perform early stopping and save computation time by defining convergence conditions. 

At regular intervals (this interval being called a *sampling period*), the agents (spin vectors) are sampled and compared with their previous state by comparing their Ising energy. If the energy is the same, the stability period of the agent is increased. If an agent's stability period exceeds a *convergence threshold*, it is considered to have converged and its state is frozen. If all agents converge before the maximum number of iterations has been reached, the algorithm then stops earlier.

The purpose of sampling the spins at regular intervals is to decorrelate them and make their stability more informative about their convergence (because the evolution of the spins is *slow* it is expected that most of the spins will not change from a time step to the following).

> The sampling period and convergence threshold are respectively set using the `sampling_period` and `convergence_threshold` parameters. Whether or not to use early-stopping is decided with the `use_window` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", use_window=True, sampling_period=30, convergence_threshold=100)
> ```

### Combining stopping criteria

Between 1 and 3 stopping criteria can be used at the same time to control the algorithm behavior. This means that the stooping condition of the algorithm can combine restrictions in terms of number of iterations and computation time, alongside early-stopping.

> ⚠️ At least one stopping criterion must be provided, i.e. at least one of `max_steps` or `timeout` must be different from `None` or `use_window` must be set to `True`. Otherwise, a `ValueError` will be raised stating: _"No stopping criterion provided."_

Any combination of `max_steps`, `timeout` and `use_window` is allowed as long as it respects the previous warning.

## Display the evolution status

This implementation offers the user the possibility to monitor the evolution of the algorithm from the terminal. For each [stopping criterion](#stopping-criteria), a progress bar can be displayed to track the current state of the optimizer:

```
🔁 Iterations       :  29%|███████████████▊                                      | 2919/10000 [00:07<00:16, 429.21 steps/s]
⏳ Simulation time  :  11%|████████▋                                                                   | 6.89/60.00 seconds
🏁 Bifurcated agents:   2%|▉                                                          | 2/128 [00:07<07:27,  3.55s/ agents]
```

Note that each progress bar is displayed only if the associated stopping criterion is being used by the SB optimizer, i.e.:
- the **🔁 Iterations** progress bar is displayed if [`max_steps`](#number-of-iterations) is not `None`
- the **⏳ Simulation times** progress bar is displayed if [`timeout`](#computation-timeout) is not `None`
- the **🏁 Bifurcated agents** progress bar is displayed if [`use_window`](#early-stopping) is set to `True`

> Whether or not to display the progress bars is set using the `verbose` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", verbose=True) # Display evolution status
> sb.minimize(polynomial, domain="spin", verbose=False) # Hide evolution status
> ```

## Outputs

## Available routines

```{eval-rst}
.. autofunction:: simulated_bifurcation.optimize
.. autofunction:: simulated_bifurcation.minimize
.. autofunction:: simulated_bifurcation.maximize
```

## Warnings

### Approximation algorithm

The Simulated Bifurcation algorithm is an approximation algorithm, which implies that the returned values may not correspond to global optima. Besides, if some constraints are embedded as penalties in the quadratic models, that is adding terms that ensure that any global optimum satisfies the constraints, the return values may violate these constraints.

### Non-deterministic behaviour

The Simulated Bifurcation algorithm uses a randomized initialization, and this package is implemented with a PyTorch backend. To ensure a consistent initialization when running the same script multiple times, use `torch.manual_seed`. However, results may not be reproducible between CPU and GPU executions, even when using identical seeds. Furthermore, certain PyTorch operations are not deterministic. For more comprehensive details on reproducibility, refer to the PyTorch documentation available at [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html).

### Liability

The authors of this package cannot be held responsible for its use. The Simulated Bifurcation algorithm can be used as a decision aid, but its results in no way replace the user's final decision. Any use of this package for commercial purposes or involving money is the sole responsibility of the user.

[^1]: Hayato Goto et al., "Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5, eaav2372(2019). DOI:10.1126/sciadv.aav2372

[^2]: Hayato Goto et al., "High-performance combinatorial optimization based on classical mechanics". Sci. Adv.7, eabe7953(2021). DOI:10.1126/sciadv.abe7953

[^3]: Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal fluctuation". Commun Phys 5, 153 (2022). https://doi.org/10.1038/s42005-022-00929-9
