# Simulated Bifurcation for Python

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
![Status](https://github.com/bqth29/simulated-bifurcation-algorithm/actions/workflows/config.yml/badge.svg)
[![codecov](https://codecov.io/gh/bqth29/simulated-bifurcation-algorithm/branch/main/graph/badge.svg?token=J76VVHPGVS)](https://codecov.io/gh/bqth29/simulated-bifurcation-algorithm)
![GitHub stars](https://img.shields.io/github/stars/bqth29/simulated-bifurcation-algorithm.svg?style=social&label=Star)

The **Simulated Bifurcation** (SB) algorithm is a fast and highly parallelizable state-of-the-art algorithm for combinatorial optimization inspired by quantum physics and spins dynamics. It relies on Hamiltonian quantum mechanics to find local minima of **Ising** problems. The last accuracy tests showed a median optimality gap of less than 1% on high-dimensional instances.

This open-source package is built upon **PyTorch** to allow GPU computations that take advantage of the great possibility of parallelization the SB algorithm, leading to high time performances that can outperform commercial solvers.

It also provides an API to define Ising models or other NP-hard and NP-complete problems (QUBO, Karp problems, ...) that can be solved using the SB algorithm.

## ⚙️ Install

```
pip install simulated-bifurcation
```

## 🧪 Ising problem

An Ising problem, given a null-diagonal square symmetrical matrix $J$ of size $N \times N$ and a vector $h$ of size $N$, consists in finding the spin vector $\mathbf{s} = (s_{1}, ... s_{N})$ called the *ground state*, (each $s_{i}$ being equal to either 1 or -1) such that the following value, called *Ising energy*, is minimal:

$$- \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij}s_{i}s_{j} + \sum_{i=1}^{N} h_{i}s_{i}$$

This problem is known to be NP-hard but is very useful since it can be used in many sectors such as finance, transportation or chemistry or derived as other well-know optimization problems (QUBO, MAXCUT, Knapsack problem, etc.).

> In the quantum theory behind the SB algorithm, $J$ must be symmetrical with a null diagonal. However, this package handles any square matrix by creating a new symmetrical matrix with null diagonal leading to an equivalent ising problem.

You can, create an Ising model instance as follow:


```python
import torch
import simulated_bifurcation as sb


J = torch.Tensor([[0, 1, 2], [1, 0, -2], [2, -2, 0]])
h = torch.Tensor([-1, 0, 2])
ising = sb.Ising(J, h)
```

> If you only wish to use the `J` matrix and not the `h` vector, you can simply call `sb.Ising(J)`.

This Ising model can then be called with spin tensors to compute its energy:

```python
# Single spin vector
spins = torch.Tensor([1, 1, -1])
ising(spins)
>>> -4.0

# Spin tensor (column-wise)
spins = torch.Tensor(
    [
        [1, -1, 1, -1],
        [-1, 1, 1, -1],
        [-1, -1, 1, -1]
    ]
)
ising(spins)
>>> [2.0, -4.0, 0.0, -2.0]
```

## 💻 The *Simulated Bifurcation* (SB) algorithm

### Behavior

### Usage on Ising instances

Ising instances can be optimized with the Simulated Bifurcation algorithm using the `optimize` method. Instances are updated in place and the results of the optimization can be retrieved by calling `ising.ground_state` (best vector) and `ising.energy` (best objective value).

```python
ising.optimize()
ising.energy
>>> -4.0

ising.ground_state
>>> Tensor([-1.0, 1.0, -1.0])
```

### Parallelization (multi-agent search)

The Simulated Bifurcation algorithm is highly parallelizable since it only relies on linear matrices equations. To take advantage of this property, this implementation offers the possibility to perform a multi-agent search of the optimal solution by evolving several spin vectors in parallel (each one being called an **agent**). The number of agents is set by the `agents` parameter in the `optimize` method.

> **💡 Tip:** it is faster to run once the algorithm with N agents than to run N times the algorithm with only one agent.

```python
# Efficient computation ✔️
ising.optimize(agents=100)

# Slower cumbersome computation ❌
for _ in range(100):
    ising.optimize(agents=1)
```

### GPU computation

This parallelization of the algorithm can also be utilized by performing calculations on GPUs to speed them up significantly. To do this, simply specify the calculation device `cuda` when instantiating an Ising model:

```python
ising = sb.Ising(J, h, device='cuda')
ising.optimize()
```

### Early stopping

The Simulated Bifurcation algorithm stops after a certain number of iterations, defined by the parameter `max_steps` of the `optimize` method. However, this implementation comes with the possibility to perform early stopping and save computation time by defining convergence conditions. 

At regular intervals, the state of the spins is sampled and compared with its previous value to calculate their stability period. If an agent's stability period exceeds a convergence threshold, it is considered to have converged and its value is frozen. If all agents converge before the maximum number of iterations has been reached, the algorithm stops.

> - The sampling period and the convergence threshold are respectively set using the `sampling_period` and `convergence_threshold` parameters of the `optimize` method.
> - To use early stopping in the SB algorithm, set the `use_window` parameter to `True`.
> - If only some agents have converged when the maximum number of iterations is reached, the algorithm stops and only the converged agents are considered in the results.

```python
# Stop with maximal iterations
ising.optimize(max_steps=10000)

# Early stopping
ising.optimize(
    sampling_period=30,
    convergence_threshold=50,
    use_window=True
)
```

### Usage on polynomial instances

The SB algorithm generalizes to a wider range of problems that can be written as the minimization or maximization of multivariable polynomials of degree two, i.e. written as

$$\sum_{i=1}^{N} \sum_{j=1}^{N} M_{ij}x_{i}x_{j} + \sum_{i=1}^{N} v_{i}x_{i} + c$$

for which the $x_{i}$'s can be spins, binary or non-negative integer.

This can also be seen as the sum of a quadratic form, a linear form and a constant term and such such a formulation is the basis of many optimization problems.

The `minimize` and `maximize` functions allow to respectively minimize and maximize the value of such polynomials for a given type of input values, relying on the SB algorithm. 

> For integer values, the number of bits with which the numbers are encoded must be specified. For instance, 9-bits integer correspond to the `'int9'` input type.

```python
matrix = torch.Tensor([[0, 1, 2], [1, 0, -2], [2, -2, 0]])
vector = torch.Tensor([-1, 0, 2])
constant = 2.0
```

#### Minimization

```python
# Spin minimization
spin_value, spin_vector = sb.minimize(matrix, vector, constant, input_type='spin')

# Binary minimization
binary_value, binary_vector = sb.minimize(matrix, vector, constant, input_type='binary')

# 3-bits integer minimization
int_value, int_vector = sb.minimize(matrix, vector, constant, input_type='int3')
```

#### Maximization

```python
# Spin maximization
spin_value, spin_vector = sb.maximize(matrix, vector, constant, input_type='spin')

# Binary maximization
binary_value, binary_vector = sb.maximize(matrix, vector, constant, input_type='binary')

# 10-bits integer maximization
int_value, int_vector = sb.maximize(matrix, vector, constant, input_type='int10')
```

> As for the Ising model, only the matrix is mandatory and the vector and the constant value are optional

The `minimize` and `maximize` functions both inherit from the `optimize` method of the `Ising` class. Thus, the early stopping (`sampling_period`, `convergence_thershold`, `max_steps`, `use_window`), multi-agent search (`agents`) and GPU calculation (`device`) parameters presented earlier can also be used in both these functions.

```python
sb.minimize(matrix, vector, constant, input_type='spin',
    sampling_period=30, convergence_thershold=50, max_steps=10000, use_window=True,
    agents=128, device='cuda')
```

## 💡 Advanced usages

This section deals with a more complex use of the SB algorithm, as it is closer to the quantum theory from which it is derived. To better understand the significance of the subjects at stake, we recommend reading the theory behind the SB algorithm by Goto *et al.*.

- *Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems* on [Science Advances](https://www.science.org/doi/pdf/10.1126/sciadv.aav2372);
- *Simulated bifurcation assisted by thermal fluctuation* on [Nature](https://www.nature.com/articles/s42005-022-00929-9.pdfhttps://www.nature.com/articles/s42005-022-00929-9.pdf);
- *High-performance combinatorial optimization based on classical mechanics* on [Science Advances](https://www.science.org/doi/full/10.1126/sciadv.abe7953).

### SB Algorithm modes

The SB algorithm is available in four different versions (Goto *et al.*) that result in small variations in the algorithm general operation. The four modes are:

1. **Ballistic SB (bSB)**: uses the particles' position for the SB matrix computations; usually slower but more accurate.
2. **Discrete SB (dSB)**: uses the sign of the particles' position for the SB matrix computations; usually fasyer but less accurate.
3. **Heated ballistic SB (HbSB)**: uses the bSB algorithm with a supplementary non-symplectic term to allow a higher solution space exploration.
4. **Heated discrete SB (HdSB)**: uses the dSB algorithm with a supplementary non-symplectic term to allow a higher solution space exploration.

These mode can be selected setting the parameters `ballistic` and `heat` to either `True` or `False` in the `Ising.optimize` method or the `minimize`/`maximize` functions.

```python
ising.optimize(ballistic=True, heat=False) #bSB
ising.optimize(ballistic=False, heat=True) #HdSB

sb.minimize(ballistic=False, heat=False) #dSB
sb.maximize(ballistic=True, heat=True) #HbSB
```

### SB Algorithm's hyperparameters setting

The SB algorithm has a set of hyperparameters corresponding to physical constants derived from quantum theory, which have been fine-tuned (Goto *et al.*) to give the best results most of the time. However, depending on the properties of the instances used, different hyperparameters may be of interest. For this purpose, the A function can be used to modify their value.

```python
import simulated_bifurcation as sb

# Custom hyperparameters values
sb.set_env(time_step=.1, pressure_slope=.01, heat_coefficient=.06)

# Default hyperparameters values
sb.reset_env()
```

### Custom models

## 🧬 Other optimizable models

Ising's model can be applied to a wide range of NP-complete and NP-hard problems, which can be solved by the SB algorithm.

> RETURN vector ?

### Second Order Unconstrained Polynomial Spin Optimization

$$\sum_{i=1}^{N} \sum_{j=1}^{N} M_{ij}s_{i}s_{j} + \sum_{i=1}^{N} v_{i}s_{i} + c$$

```python
import torch
from simulated_bifurcation.models import Spin
```

```python
Q = torch.Tensor(
    [
        [1, 2, -3],
        [0, -4, 5],
        [0, 0, 6]
    ]
)

qubo = Spin(M, v, c)
qubo.optimize()

qubo.best_binary_vector
>>> torch.Tensor([0., 1., 0.])
qubo.best_objective_value
>>> -4.0
```

### Quadratic Unconstrained Binary Optimization (QUBO)

The QUBO problem consists in minimizing a quadratic form defined by an upper-triangular matrix $Q$ on the set of the binary vectors. This means finding the vector $\bold{x} = (x_{1}, ..., x_{N})$ with all $x_{i}$'s being either 0 or 1 such that the following value is minimal:

$$\sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij}x_{i}x_{j}$$

A QUBO problem can easily be reformulated as an equivalent Ising model which makes it suitable to be opitmized by the SB algorithm.

#### Usage

```python
import torch
from simulated_bifurcation.models import QUBO
```

```python
Q = torch.Tensor(
    [
        [1, 2, -3],
        [0, -4, 5],
        [0, 0, 6]
    ]
)

qubo = QUBO(Q)
qubo.optimize()

qubo.best_binary_vector
>>> torch.Tensor([0., 1., 0.])
qubo.best_objective_value
>>> -4.0
```

### Second Order Unconstrained Polynomial Binary Optimization 

The QUBO problem can also be extended to second-order polynomial binary optimization by adding a linear form and a constant term to the equation, and can still be handled by the SB algorithm:

$$\sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij}x_{i}x_{j} + \sum_{i=1}^{N} l_{i}x_{i} + c$$

#### Usage

```python
import torch
from simulated_bifurcation.models import Binary
```

```python
Q = torch.Tensor(
    [
        [1, 2, -3],
        [0, -4, 5],
        [0, 0, 6]
    ]
)
l = torch.Tensor([-1, 0, 2])
c = -5

binary = Binary(Q, l, c)
binary.optimize()

binary.best_binary_vector
>>> torch.Tensor([0., 1., 0.])
binary.best_objective_value
>>> -9.0
```

### Second Order Unconstrained Polynomial Integer Optimization 

$$\sum_{i=1}^{N} \sum_{j=1}^{N} M_{ij}n_{i}n_{j} + \sum_{i=1}^{N} v_{i}n_{i} + c$$

### `IsingInterface` API

More generally, the Ising model can be adapted to represent a wider range of NP-hard and NP-complete problems, including the [21 Karp problems](https://arxiv.org/pdf/1302.5843.pdf). To allow the use of the SB algorithm on various problems, this package provides an API to easily transform NP-hard and NP-complete problems to Ising models on which the SB algorithm can be used.

---

_Simulated bifurcation_ is a state-of-the-art algorithm based on quantum physics theory and used to approximize very accurately and quickly the optimal solution of Ising problems. 
> You can read about the scientific theories at stake and the engineering of the algorithm by Goto *et al.* here: https://www.nature.com/articles/s42005-022-00929-9

Ising problems can be used in many sectors such as finance, transportation or chemistry or derived as other well-know optimization problems (QUBO, Knapsack problem, ...).

## 🚀 Optimization of an Ising model

### Definition

An Ising problem, given a **square symmetric** matrix `J` of size `n x n` and a vector `h` of size `n`, consists in finding the spin vector `s = [s_1, s_2, ..., s_n]`, called the *ground state*, (each `s_i` is either `1` or `-1`) such that the value `E = - 0.5 * ∑∑ J_ij*s_i*s_j + ∑ h_i*s_i`, called *Ising energy*, is minimal.

### Create an instance

Given `J` and `h`, creating an Ising model is done as follows:

```python
import simulated_bifurcation as sb

ising = sb.Ising(J, h)
```

### Optimize a model

To optimize an Ising model using the Simulated Bifurcation algorithm, you simply need to run:

```python
ising.optimize()
```

The `optimize` methods takes several parameters that are presented in the dedicated section.

### Retrieve the ground state / Ising energy

Once the model is optimized, you can get the best found Ising features using model's attributes

```python
ising.energy # Ising energy -> float
ising.ground_state # Ground state (best spin vector) -> torch.Tensor
```

## 📊 Optimization parameters

The `optimize` methods uses a lot of parameters but only some of them may be changes since the biggest part has been set after reserach and fine-tuning work.

### Quantum parameters

These parameters stem from the quantum theory Their purpose is described in the paper cited above.

> The parameters marked with ⚠️ should not be changed to ensure a good accuracy of the algorithm.

- `pressure_slope` ⚠️
- `gerschgorin`: if `True` then uses the Gerschgorin's theorem to set the scale value; else uses the uses the value defined by Goto *et al.*
- `heat_parameter` ⚠️
- `time_step` ⚠️

### Simulated Bifurcation modes

There are four modes of the algorithm (ballistic v. discrete + heated v. non-heated) that result in small variations in the algorithm general operation. These mode can be selected setting the parameters `ballistic` and `heated` to `True` or `False`.

> The ballistic mode is supposed to give a slighter less satisfying accuracy but to converge faster in comparison to the discrete mode which is generally more accurate but also a bit slower.

### Early stopping

One particularity of our implementation of the Simulated Bifurcation algorithm is the possibility to perform an early stopping and save computation time. The sampling frequence and window size for deciding whether to stop or continue can be set through the parameters `sampling_period` and `convergence_threshold`. 

> Yet, the default parameters have been set as the result of a good trade-off betwwen computation time and accurary so it is not recommanded to change them.

To use early stopping, the `use_window` parameter must be set to `True`. Both ways, the algorithm will stop after a certain number of iterations (if early stopping conditions were not met or if `use_window` was set to `False`) that is defined by the `max_steps` parameter.

### Multi-agent optimization

This version of the Simulated Bifurcation algorithm also allows a multi-agent search of the optimal solution which benefits from the parallelization of the computations. The number of agents is set by the `agents` parameter.

> **💡 Tip:** it is faster to run once the algorithm with N agents than to run N times the algorithm with only one agent.

### Displaying the state of evolution

Finally, you can choose to show or hide the evolution of the algorithm setting the `verbose` parameter to either `True` or `False`.

> If you choose to set `verbose = True`, the evolution will be displayed as `tqdm` progress bar(s) in your terminal.

## 🔀 Derive the algorithm for other problems using the IsingInterface API

A lot of mathematical problems ([QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization), [TSP](https://en.wikipedia.org/wiki/Travelling_salesman_problem), ...) can be written as Ising problems, and thus can be solved using the Simulated Bifurcation algorithm. Some of them are already implemented in the `models` folder but you are free to create your own models using our API.

To do so, you need to create a subclass of the abstract class `IsingInterface` present in the `interface` submodule. The `IsingInterface` has only two attributes `dtype` and `device` which allow you to set the type of the data and the device with which you wish to work.

```python
from simulated_bifurcation.interface import IsingInterface


class YourModel(IsingInterface):

    def __init__(self, dtype, device, *args, **kwargs):
        super().__init__(dtype, device) # Mandatory
        # YOUR CODE HERE
        ...
```

Once created, such an object can be optimized using the same principle as an `Ising` object, using the `optimize` methods which uses the same parameters as the `Ising`'s one:

```python
your_model = YourModel(...)
your_model.optimize()
```

Yet, to make it work, you will first have to overwrite two abstract methods of the `IsingInterface` class (`to_ising` and `from_ising`) that are called by the `optimize` method. Otherwise you will get a `NotImplementedError` error message.

When the `optimize` method is called, an equivalent Ising model will first be created using `to_ising` and then optimized using the exact same parameters you provided as input for the `IsingInterface.optimize` method. Once it is optimized, information for your own model will be derived from the optimal features of this equivalent Ising model using `from_ising`.

### `to_ising` method

The `to_ising` is meant to create an instance of an Ising model based on the data of your problem. It takes no argument and must only return an `Ising` object. The idea is to rely on the parameters of your problems to derive an Ising representation of it. At some point in the definition of the method, you will have to create the `J` matrix and the `h` vector and eventually return `Ising(J, h)`.

```python
def to_ising(self) -> sb.Ising:
    # YOUR CODE HERE
    J = ...
    h = ...
    return sb.Ising(J, h, dtype=self.dtype, device=self.device)
```

> Do not forget to set the `device` attribute when you instantiate the class if you are working on a GPU because all the tensors must be set on the same device.

### `from_ising` method

The `from_ising` is the reciprocal method. Once the equivalent Ising model of your problem has been optimized, you can retrieve information from its ground state and/or energy and adapt them to your own problem. It must only take an `Ising` object for input and return `None`.

```python
def from_ising(self, ising: sb.Ising) -> None:
    # YOUR CODE HERE
    return 
```

### Binary and integer formulations

Note that many problems that can be represented as Ising models are not based on spin vectors but rather on binary or integer vectors. The `interface` submodule thus has two additional classes, `Binary` and `Integer`, both of which inherit from `IsingInterface` in order to generalize these cases more easily.

> 🔎 You can check [Andrew Lucas' paper](https://arxiv.org/pdf/1302.5843.pdf) on Ising formulations of NP-complete and NP-hard problems, including all of Karp's 21 NP-complete problems.

## 🔗 Cite this work

If you are using this code for your own projects please cite our work:

```bibtex
@software{Ageron_Simulated_Bifurcation_SB_2022,
    author = {Ageron, Romain and Bouquet, Thomas and Pugliese, Lorenzo},
    month = {4},
    title = {{Simulated Bifurcation (SB) algorithm for Python}},
    version = {1.2.0},
    year = {2023}
}
```
