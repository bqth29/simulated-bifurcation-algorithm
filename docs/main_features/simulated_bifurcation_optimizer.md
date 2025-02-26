# Simulated Bifurcation Optimizer

<!--
    TODO: update with the latest versions of the parameters
    - `domain` will move from `str` to `Union[str, List[str]]` for multi-domain optimization
    - `ballistic` will be renamed and its value will be a `Literal` (either `"ballistic"` or `"discrete"`)
-->

This package provides a Python implementation of all four versions of the Simulated Bifurcation algorithm (bSB, dSB, HbSB or HdSB) enhanced with supplementary features that allow the user, for instance, to define advanced stop criteria and harness the parallelization of the algorithm by running a multi-agent optimization CPU or GPU.

Three optimization functions (`optimize`, `maximize` and `minimize`) that all share the same pool of parameters are available. These parameters are meant to tune the different extra-features of the algorithm and to allow a more personalized experience. They are gathered in the following table and their usage and specific role in the optimizer are described thoughout this page.

> - The mandatory parameters are written in **bold**
> - Except for `polynomial` which is positional and must be defined as the first parameter, all other parameters are keyword-only and their order does not matter
> - `optimize` has an extra `minimize` boolean parameter (default `True`)
> - The parameters are for optimization features only, for parameters related to quantum physics theory, see [Advanced Usage](advanced_usage.md)

| Parameter                                       | Type                         | Default value   | Usage                                                                                                                                                                                  |
| ----------------------------------------------- | ---------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**`polynomial`**](#quadratic-model-definition) | `PolynomialLike`             |                 | Quadratic model to optimize.                                                                                                                                                           |
| [`agents`](#parallelization-multi-agent-search) | `int`                        | `128`           | Number of oscillators to evolve in parallel                                                                                                                                            |
| [`mode`](#sb-algorithm-versions)                | `"ballistic` or `"discrete"` | `True`          | Whether to use the ballistic version of the SB algorithm (bSB) or the discrete version (dSB).                                                                                          |
| [`best_only`](#outputs)                         | `bool`                       | `True`          | Whether to only return the best agent and its associated objective function value, or all agents at once.                                                                              |
| [`convergence_threshold`](#early-stopping)      | `int`                        | `50`            | Number of consecutive samplings after which an agent is considered to have converged if its Ising energy has not changed. Its value is read only if `early_stopping` is set to `True`. |
| [`device`](#gpu-computation)                    | `str` or `torch.device`      | `None`          | Device on which to run the optimization (CPU or GPU).                                                                                                                                  |
| [**`domain`**](#optimization-domain)            | `str`                        |                 | Domain on which the optimization is carried out (spin, binary or integer values).                                                                                                      |
| [`dtype`](#quadratic-model-definition)          | `torch.dtype`                | `torch.float32` | Computation dtype.                                                                                                                                                                     |
| [`early_stopping`](#early-stopping)             | `bool`                       | `True`          | Whether to use early-stopping or not.                                                                                                                                                  |
| [`heated`](#sb-algorithm-versions)              | `bool`                       | `False`         | Whether to use the heated version of the SB algorithm or not.                                                                                                                          |
| [`max_steps`](#number-of-iterations)            | `int`                        | `10000`         | Maximum number of iterations of the optimizer (one iteration is one step in the symplectic Euler scheme). If reached, the computation is stopped and the current results are returned. |
| [`sampling_period`](#early-stopping)            | `int`                        | `50`            | Number of iterations between two successive oscillator samplings to verify early stopping conditions. Its value is read only if `early_stopping` is set to `True`.                     |
| [`timeout`](#computation-timeout)               | `int`                        | `None`          | Maximum computation time of the optimizer in seconds. If reached, the computation is stopped and the current results are returned.                                                     |  |
| [`verbose`](#display-the-evolution-status)      | `bool`                       | `True`          | Whether to display the evolution status of the optimizer with progress bars or not.                                                                                                    |

## Quadratic model definition

The quadratic model to optimize can be defined in a standalone mode with the `build_model` function of the package as presented in the [Quadratic Models](quadratic_models.md) page. Creating a model this way is useful if you intend to use it for other purposes like evaluating input data or defining a custom model dtype.

When a model is created with the `build_model` function, you can optimize it using one of the three optimization methods (`optimize`, `maximize` and `minimize`) of the `QuadraticPolynomial` class. These methods have the same name and are equivalent to the three optimization (static) functions described in this page. The only difference is that the methods of the `QuadraticPolynomial` class have no positional argument `polynomial`.

> Not paying attention to the difference between the model dtype and the computation dtype which is discussed [further on this page](#computation-dtype), it is equivalent to use:
> ```python
> sb.minimize(polynomial, **kwargs)
> ```
> and
> ```python
> sb.build_model(polynomial).minimize(**kwargs)
> ```
> Note that the same holds for `maximize` and `optimize`.

However, if you want to minimize a model "*in one go*" without having to reuse it afterwards, you can define it directly in the optimization function (`optimize`, `maximize` or `minimize`). The model can be defined using [tensors](quadratic_models.md#using-tensors) or a [SymPy expression](quadratic_models.md#using-a-sympy-expression) as explained in the same [Quadratic Models](quadratic_models.md) page.

> The optimization model is passed as the only positional argument(s) of the optimization method.

## Optimization domain

Once the model is defined, the optimization domain must be set. It corresponds to the space of values that is searched by the Simulated Bifurcation algorithm to find the optimal values of decision variables. There are three possible types of domains on which the optimization can be carried out: spin (-1 and 1), binary (0 and 1) and integer.

> The optimization domain is set using the `domain` parameter. For spin and binary optimization, the domain must respectively be `"spin"` or `"binary"`. For integer optimization, the domain must start with `"int"` followed by a positive integer which indicates the number of bits to represent the integer values. More formally, it must match the regular expression `^int[1-9][0-9]*$`. For instance, `"int2"` represents all the integer that can be incoded on 2 bits, i.e. 0, 1, 2 and 3. Similarly, `"int10"` represents all the integer that can be incoded on 10 bits, i.e. all integers between 0 and 1023.
>
> ```python
> sb.minimize(polynomial, domain="spin") # Optimization on {-1, 1}
> sb.minimize(polynomial, domain="binary") # Optimization on {0, 1}
> sb.minimize(polynomial, domain="int3") # Optimization on {0, 1, 2, 3, 4, 5, 6, 7}
> ```

## Computation dtype

It is possible to configure the computation dtype of the algorithm. On the one hand using a dtype with a less bits can improve the algorithm performances speed-wise and on the other hand, a dtype with a heavier bit representation ay be more accurate. The Simulated Bifurcation is based on numerical values betwwen -1 and 1 thus, only `torch.float32` and `torch.float64` are currently available. Because some key PyTorch methods used in the Simulated Bifurcation backend are not available for `torch.float16`, this dtype cannot be used to run the computations.

Note that the computation dtype is only used for backend computations. The optimization model, if defined in a standalone mode, can have a different dtype (see how to build an optimization model in the [Quadratic Models](quadratic_models.md) page). However, when calling the SB algorithm directly with one of the three `optimize`, `maximize` and `minimize` functions, the computation dtype is also used as the model' dtype.

> The computation dtype is set using the `dtype` parameter:
>
> ```python
> sb.minimize(polynomial, dtype=torch.float32, **kwargs)
> ```

If you want to run the SB algorithm using the `torch.float32` dtype and want to create an integer model at the same time, for instance using the `torch.float8` dtype, the model should be created first and the optimization method called from this model in a second time:

```python
model = sb.build_model(polynomial, dtype=torch.int8) # The dtype parameter refers to the model's dtype
model.minimize(dtype=torch.float32, **kwargs) # The dtype parameter refers to the computation dtype
```

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

> The sampling period and convergence threshold are respectively set using the `sampling_period` and `convergence_threshold` parameters. Whether or not to use early-stopping is decided with the `early_stopping` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", early_stopping=True, sampling_period=30, convergence_threshold=100)
> ```

### Combining stopping criteria

Between 1 and 3 stopping criteria can be used at the same time to control the algorithm behavior. This means that the stooping condition of the algorithm can combine restrictions in terms of number of iterations and computation time, alongside early-stopping.

> ⚠️ At least one stopping criterion must be provided, i.e. at least one of `max_steps` or `timeout` must be different from `None` or `early_stopping` must be set to `True`. Otherwise, a `ValueError` will be raised stating: _"No stopping criterion provided."_

Any combination of `max_steps`, `timeout` and `early_stopping` is allowed as long as it respects the previous warning.

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
- the **🏁 Bifurcated agents** progress bar is displayed if [`early_stopping`](#early-stopping) is set to `True`

> Whether or not to display the progress bars is set using the `verbose` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", verbose=True) # Display evolution status
> sb.minimize(polynomial, domain="spin", verbose=False) # Hide evolution status
> ```

## Outputs

Two tensors are returned by the optimization function: the optimized agents (with values in the optimization domain) and their evaluation by the input polynomial.

> The returned agents' dtype is the same as the model's dtype.

Users may also want to only get the best computed agent and its associated evalutation so this is configurable when calling the optimization function. The notion of *best* agents means the agent with the highest evaluation for a maximization or with the lowest evaluation for a minimization.

> Whether or not to return only the best agent is set using the `best_only` parameter:
>
> ```python
> sb.minimize(polynomial, domain="spin", best_only=True) # Only return the best agent and its evaluation
> sb.minimize(polynomial, domain="spin", best_only=False) # Return all agents and their evaluations
> ```

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
