# Simulated Bifurcation for Python

Python implementation of the _Simulated Bifurcation_ algorithm in order to approximize the optimal solution of Ising problems. The last accuracy tests showed a median optimality gap of less than 1% on high-dimensional instances.

## ⚙️ Install

1. Clone the repository

```
git clone https://github.com/bqth29/simulated-bifurcation-algorithm.git
```

1. Change directory

```
cd simulated-bifurcation-algorithm
```

1. Intall dependencies

```powershell
python -m pip install -r requirements.txt
```

## 🧪 Scientific background

_Simulated bifurcation_ is a state-of-the-art algorithm based on quantum physics theory and used to approximize very accurately and quickly the optimal solution of Ising problems. 
>You can read about the scientific theories at stake and the engineering of the algorithm here: https://arxiv.org/abs/2108.03092

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
ising.ground_state # Ground state (best spin vector) -> numpy.ndarray
```

## 📊 Optimization parameters

*Coming soon*

## 🔀 Derive the algorithm for other problems using the SBModel API

A lot of mathematical problems can be written as Ising problems, and thus can be solved using the Simulated Bifurcation algorithm. Some of them are already implemented in the `models` folder but you are free to create your own models using our API.

To do so, you need to create a subclass of the abstract class `SBModel`.

```python
class YourModel(sb.SBModel):
    
    ...
```

Once created, such an object can be optimized using the same principle as an `Ising` object, using the `optimize` methods which uses the same parameters as the `Ising`'s one:

```python
your_model = YourModel(...)
your_model.optimize()
```

Yet, to make it work, you will first have to overwrite two abstract methods of the `SBModel` class (`__to_Ising__` and `__from_Ising__`) that are called by the `optimize` method. Otherwise you will get a `NotImplementedError` error message.

When the `optimize` method is called, an equivalent Ising model will first be created using `__to_Ising__` and then optimized using the exact same parameters you provided as input for the `SBModel.optimize` method. Once it is optimized, information for your own model will be derived from the optimal features of this equivalent Ising model using `__from_Ising__`.

### `__to_Ising__` method

The `__to_Ising__` is meant to create an instance of an Ising model based on the data of your problem. It takes no argument and must only return an `Ising` object. The idea is to rely on the parameters of your problems to derive an Ising representation of it. At some point in the definition of the method, you will have to create the `J` matrix and the `h` vector and eventually return `Ising(J, h)`.

```python
def __to_Ising__(self) -> sb.Ising:
    # YOUR CODE HERE
    J = ...
    h = ...
    return sb.Ising(J, h)
```

### `__from_Ising__` method

The `__from_Ising__` is the reciprocal method. Once the equivalent Ising model of your problem has been optimized, you can retrieve information from its ground state and/or energy and adapt them to your own problem. It must only take an `Ising` object for input and return `None`.

```python
def __from_Ising__(self, ising: sb.Ising) -> None:
    # YOUR CODE HERE
    return 
```

## 🔗 Cite this repository

If you are using this code for your own projects please cite our work:

```bibtex
@software{Ageron_Simulated_Bifurcation_SB_2022,
    author = {Ageron, Romain and Bouquet, Thomas and Pugliese, Lorenzo},
    month = {9},
    title = {{Simulated Bifurcation (SB) algorithm for Python}},
    version = {2.0.1},
    year = {2022}
}
```
