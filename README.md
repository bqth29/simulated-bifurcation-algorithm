# Simulated Bifurcation for Python

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyPI package](https://badge.fury.io/py/simulated-bifurcation.svg)](https://pypi.org/project/simulated-bifurcation/)
[![codecov](https://codecov.io/gh/bqth29/simulated-bifurcation-algorithm/branch/main/graph/badge.svg?token=J76VVHPGVS)](https://codecov.io/gh/bqth29/simulated-bifurcation-algorithm)
![Status](https://github.com/bqth29/simulated-bifurcation-algorithm/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/simulated-bifurcation-algorithm/badge/?version=latest)](https://simulated-bifurcation-algorithm.readthedocs.io/en/latest/?badge=latest)
![GitHub stars](https://img.shields.io/github/stars/bqth29/simulated-bifurcation-algorithm.svg?style=social&label=Star)

The **Simulated Bifurcation** (SB) algorithm is a fast and highly parallelizable state-of-the-art algorithm for combinatorial optimization inspired by quantum physics and spins dynamics. It relies on Hamiltonian quantum mechanics to find local minima of **Ising** problems. The last accuracy tests showed a median optimality gap of less than 1% on high-dimensional instances.

This open-source package utilizes **PyTorch** to leverage GPU computations, harnessing the high potential for parallelization offered by the SB algorithm.

It also provides an API to define Ising models or other NP-hard and NP-complete problems (QUBO, Karp problems, ...) that can be solved using the SB algorithm.

## ⚙️ Install

<table>
<thead>
<tr>
<th>Compute Plateform</th>
<th>CPU</th>
<th>GPU</th>
</tr>
</thead>
<tbody>
<tr>
<th>Instructions</th>
<td>

```console
pip install simulated-bifurcation     
```

</td>
<td>

&nbsp;&nbsp;&nbsp;
Install [PyTorch](https://pytorch.org/get-started/locally/) with GPU support

```console
pip install simulated-bifurcation     
```

</td>
</tr>
</tbody>
</table>

## 🧪 The _Simulated Bifurcation_ (SB) algorithm

### Ising model

An Ising problem, given a null-diagonal square symmetrical matrix $J$ of size $N \times N$ and a vector $h$ of size $N$, consists in finding the spin vector $\mathbf{s} = (s_{1}, ... s_{N})$ called the _ground state_, (each $s_{i}$ being equal to either 1 or -1) such that the following value, called _Ising energy_, is minimal:

$$- \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij}s_{i}s_{j} + \sum_{i=1}^{N} h_{i}s_{i}$$

This problem is known to be NP-hard but is very useful since it can be used in many sectors such as finance, transportation or chemistry or derived as other well-know optimization problems (QUBO, MAXCUT, Knapsack problem, etc.).

The Simulated Bifurcation algorithm was originally introduced to solve Ising problems by simulating the adiabatic evolution of spins in a quantum Hamiltonian system, but can also be generalized to a wider range of optimization problems.

### Usage on polynomial instances

The SB algorithm can be written as the minimization or maximization of multivariable polynomials of degree two, i.e. written as

$$\sum_{i=1}^{N} \sum_{j=1}^{N} M_{ij}x_{i}x_{j} + \sum_{i=1}^{N} v_{i}x_{i} + c$$

for which the $x_{i}$'s can be spins, binary or non-negative integer.

This can also be seen as the sum of a quadratic form, a linear form and a constant term and such a formulation is the basis of many optimization problems.

The `minimize` and `maximize` functions allow to respectively minimize and maximize the value of such polynomials for a given type of input values, relying on the SB algorithm. They both return the optimal polynomial value found by the SB algorithm, along with its associated input vector.

The input types must be passed to the `domain` argument:

- `spin` (default value) for a spin optimization: the optimal vector will only have ±1 values
- `binary` for a binary optimization: the optimal vector will only have 0 or 1 values
- `intX` for a `X`-bits encoded integer optimization: the optimal vector will only have integer value encoded with `X` bits or less, i.e. belonging to the range 0 to $2^{X} - 1$.

> For instance, 9-bits integer correspond to the `int9` input type and the accepted values span from 0 to 511.

```python
import simulated_bifurcation as sb
```

```python
matrix = torch.tensor([[1, 1, 2], [0, -1, -2], [-2, 0, 2]])
vector = torch.tensor([-1, 0, 2])
constant = 2.0
```

The package provides a `polynomial` API to build quadratic multivariate polynomials from such tensors. Four options are possible.

A polynomial can be defined using coefficient tensors or SymPy expressions to define polynomials in a more natural way from mathematical equations.

> The four following code snippets all create equivalent polynomials

1. Using the `QuadraticPolynomial` class

```python
from simulated_bifurcation.core import QuadraticPolynomial
```

**With tensors**

```python
polynomial = QuadraticPolynomial(matrix, vector, constant)
```

**With a SymPy expression**

```python
from sympy import poly, symbols
x, y, z = symbols("x y z")
expression = poly(
    x**2 - y**2 + 2 * z**2
    + x * y + 2 * x * z
    - 2 * y * z
    - 2 * z * x
    - x + 2 * z
    + 2
)

polynomial = QuadraticPolynomial(expression)
```

2. Using the `sb.build_model` function

**With tensors**

```python
polynomial = sb.build_model(matrix, vector, constant)
```

**With a SymPy expression**

```python
from sympy import poly, symbols
x, y, z = symbols("x y z")
expression = poly(
    x**2 - y**2 + 2 * z**2
    + x * y + 2 * x * z
    - 2 * y * z
    - 2 * z * x
    - x + 2 * z
    + 2
)

polynomial = sb.build_model(expression)
```

The `minimize` and `maximize` functions allow to respectively minimize and maximize the value of such polynomials for a given type of input values, relying on the SB algorithm. They both return the optimal polynomial value found by the SB algorithm, along with its associated input vector.

#### Minimization

```python
# Spin minimization
spin_value, spin_vector = sb.minimize(matrix, vector, constant, domain='spin')

# Binary minimization
binary_value, binary_vector = sb.minimize(matrix, vector, constant, domain='binary')

# 3-bits integer minimization
int_value, int_vector = sb.minimize(matrix, vector, constant, domain='int3')
```

Or, using a SymPy expression:

```python
# Spin minimization
spin_value, spin_vector = sb.minimize(expression, domain='spin')

# Binary minimization
binary_value, binary_vector = sb.minimize(expression, domain='binary')

# 3-bits integer minimization
int_value, int_vector = sb.minimize(expression, domain='int3')
```

#### Maximization

```python
# Spin maximization
spin_value, spin_vector = sb.maximize(matrix, vector, constant, domain='spin')

# Binary maximization
binary_value, binary_vector = sb.maximize(matrix, vector, constant, domain='binary')

# 10-bits integer maximization
int_value, int_vector = sb.maximize(matrix, vector, constant, domain='int10')
```

Or, using a SymPy expression:

```python
# Spin minimization
spin_value, spin_vector = sb.maximize(expression, domain='spin')

# Binary minimization
binary_value, binary_vector = sb.maximize(expression, domain='binary')

# 3-bits integer minimization
int_value, int_vector = sb.maximize(expression, domain='int10')
```

> For both functions, only the matrix is required, the vector and constant terms are optional.

### Parallelization (multi-agent search)

The Simulated Bifurcation algorithm is highly parallelizable since it only relies on linear matrices equations. To take advantage of this property, this implementation offers the possibility to perform a multi-agent search of the optimal solution by evolving several spin vectors in parallel (each one being called an **agent**). The number of agents is set by the `agents` parameter in the `minimize` and `maximize` functions.

> **💡 Tip:** it is faster to run once the algorithm with N agents than to run N times the algorithm with only one agent.

```python
# Efficient computation ✔️
sb.minimize(matrix, agents=100)

# Slower cumbersome computation ❌
for _ in range(100):
    sb.minimize(matrix, agents=1)
```

### GPU computation

This parallelization of the algorithm can also be utilized by performing calculations on GPUs to speed them up significantly. To do this, simply specify the calculation `device` argument to `cuda` when instantiating an Ising model:

```python
sb.minimize(matrix, device='cuda')
```

### Early stopping

The Simulated Bifurcation algorithm stops after a certain number of iterations, defined by the parameter `max_steps` of the `minimize` and `maximize` functions. However, this implementation comes with the possibility to perform early stopping and save computation time by defining convergence conditions.

At regular intervals, the energy of the agents is sampled and compared with its previous value to calculate their stability period. If an agent's stability period exceeds a convergence threshold, it is considered to have converged and its value is frozen. If all agents converge before the maximum number of iterations has been reached, the algorithm stops.

- The sampling period and the convergence threshold are respectively set using the `sampling_period` and `convergence_threshold` parameters of the `minimize` and `maximize` functions.
- To use early stopping in the SB algorithm, set the `early_stopping` parameter to `True`.
- If only some agents have converged when the maximum number of iterations is reached, the algorithm stops and only these agents are considered in the results.

```python
# Stop with maximal iterations
sb.minimize(matrix, max_steps=10000)

# Early stopping
sb.minimize(
    matrix,
    sampling_period=30,
    convergence_threshold=50,
    early_stopping=True,
)
```

### Optimization results

By default, SB returns the best vector and objective value found. However, it is also possible to configure it to so it returns all the vectors for each agent with the associated objective value. To do so, the `best_only` parameter of the `minimize` and `maximize` functions must be set to `False` (default is `True`).

```python
best_vector, best_value = sb.minimize(matrix, best_only=True)
vectors, values = sb.maximize(matrix, best_only=False)
```

## 💡 Advanced usages

This section deals with a more complex use of the SB algorithm, as it is closer to the quantum theory from which it is derived. To better understand the significance of the subjects at stake, we recommend reading the theory behind the SB algorithm by Goto _et al._.

- Goto, H., Tatsumura, K., & Dixon, A. R. (2019). Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems. _Science advances, 5_(4), eaav2372.
- Kanao, T., & Goto, H. (2022). Simulated bifurcation assisted by thermal fluctuation. _Communications Physics, 5_(1), 153.
- Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., ... & Tatsumura, K. (2021). High-performance combinatorial optimization based on classical mechanics. _Science Advances, 7_(6), eabe7953.

### SB Algorithm modes

The SB algorithm is available in four different versions (Goto _et al._) that result in small variations in the algorithm general operation. The four modes are:

1. **Ballistic SB (bSB)**: uses the particles' position for the SB matrix computations; usually faster but less accurate.
2. **Discrete SB (dSB)**: uses the sign of the particles' position for the SB matrix computations; usually slower but more accurate.
3. **Heated ballistic SB (HbSB)**: uses the bSB algorithm with a supplementary non-symplectic term to allow a higher solution space exploration.
4. **Heated discrete SB (HdSB)**: uses the dSB algorithm with a supplementary non-symplectic term to allow a higher solution space exploration.

These mode can be selected setting the parameters `ballistic` and `heated` to either `True` or `False` in the `Ising.optimize` method or the `minimize`/`maximize` functions.

```python
sb.minimize(matrix, ballistic=True, heated=False)  # bSB
sb.minimize(matrix, ballistic=False, heated=True)  # HdSB

sb.maximize(matrix, ballistic=False, heated=False)  # dSB
sb.maximize(matrix, ballistic=True, heated=True)  # HbSB
```

### SB Algorithm's hyperparameters setting

The SB algorithm has a set of hyperparameters corresponding to physical constants derived from quantum theory, which have been fine-tuned (Goto _et al._) to give the best results most of the time. Nevertheless, the relevance of specific hyperparameters may vary depending on the properties of the instances. For this purpose, the `set_env` function can be used to modify their value.

```python
# Custom hyperparameters values
sb.set_env(time_step=.1, pressure_slope=.01, heat_coefficient=.06)

# Default hyperparameters values
sb.reset_env()
```

### Derived optimization models

A lot of mathematical problems (QUBO, Travelling Salesman Problem, MAXCUT, ...) can be written as order-two multivariate polynomials problems, and thus can be solved using the Simulated Bifurcation algorithm. Some of them are already implemented in the `models` module:

**🔬 Physics**

- Ising model

**📐 Mathematics**

- Quadratic Unconstrained Binary Optimization (QUBO)
- Number partitioning

**💸 Finance**

- Markowitz model

### Custom models

You are also free to create your own models using our API. Depending on the type of model you wish to implement, you can create a subclass of the `ABCModel` class to quickly and efficiently link your custom model to an Ising problem and solve it using the SB algorithm. Such a model must have a `domain` class attribute that set the definition domain of all the instances.

The advantage of doing so is that your model can directly call the `optimize` method that it inherits from the `QuadraticPolynomial` interface without having to redefine it.

For instance, here is how the QUBO model was implemented:

> The QUBO problem consists, given an upper triangular matrix $Q$, in finding the binary vector that minimizes the value
> $$\sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij}x_{i}x_{j}$$

```python
from simulated_bifurcation.models import ABCModel


class QUBO(ABCModel):

    domain = "binary"

    def __init__(
        self,
        Q: Union[torch.Tensor, np.ndarray],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(Q, dtype=dtype, device=device)
        self.Q = self[2]
```

> You can check Andrew Lucas' paper on Ising formulations of NP-complete and NP-hard problems, including all of Karp's 21 NP-complete problems.
> 
> [🔎 Lucas, A. (2014). Ising formulations of many NP problems. _Frontiers in physics, 2_, 5.](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full)

## 🔗 Cite this work

If you are using this code for your own projects please cite our work:

[comment]: # (!MDC{begin}{BibTeX})

```bibtex
@software{Ageron_Simulated_Bifurcation_SB_2023,
    author = {Ageron, Romain and Bouquet, Thomas and Pugliese, Lorenzo},
    month = nov,
    title = {{Simulated Bifurcation (SB) algorithm for Python}},
    url = {https://github.com/bqth29/simulated-bifurcation-algorithm},
    version = {1.2.1},
    year = {2023},
}
```

[comment]: # (!MDC{end}{BibTeX})
