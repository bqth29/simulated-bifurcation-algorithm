# simulated-bifurcation-algorithm
Python implementation of a _Simulated Bifurcation_ algorithm in order to approximize the optimal assets allocation for a portfolio of _S&P 500_ assets.

## Install required packages
This algorithm relies on several Python packages. To install them all, execute the following command : 
```
python -m pip install -r requirements.txt
```

## Scientific background

_Simulated bifurcation_ is an all new calculation method based on quantum physics used to approximize very accurately and quickly the optimal solution of Ising problems. 

## Simulation

### Create a basic Markowitz model

The algorithm is meant to optimize a Markowitz portfolio. You can use your own data by using your own covariance matrix and expected return vector as follow:
```
>>> from models.Markowitz import Markowitz
>>> markowitz = Markowitz(covariance, expected_return)
```

The other possibility is to generate automatically these matrices using the built-in data generator from `.csv` data:

```
>>> from models.Markowitz import Markowitz
>>> markowitz = Markowitz.from_csv()
```

You can then run the optimization to get the portfolio using simulated bifurcation.

```
>>> markowitz.optimize()
>>> print(markowitz.as_dataframe())
    assets  stocks  ratios
3       AA     1.0   0.662
6     AAPL     1.0   0.662
129    ACN     1.0   0.662
4      AEP     1.0   0.662
5      AIG     1.0   0.662
..     ...     ...     ...
102      X     1.0   0.662
72     XEL     1.0   0.662
37     XOM     1.0   0.662
41    XRAY     1.0   0.662
147   ZBRA     1.0   0.662
```

### Advanced models

#### Taking risks in account

The `Markowitz` class allow you to set a risk coefficient that represents the importance of volatility in the optimization. The higher this coefficient, the lesser the algorithm will *take risks* optimizing the portfolio and tend to reduce the number of investments.

```
>>> markowitz = Markowitz(risk_coefficient = 10) # default 1.0
```

#### Binary encoding

You also have the possibility to set the number of bits on which the weights of the portfolio are encoded. This is a modeling of a budget limitation. 

```
>>> markowitz = Markowitz(number_of_bits = 3) # default 1
```

> **Remark:** note that the lower the number of bits for the weights encoding is, the higher the accuracy of the algorithm is. This is due to a higher weighting of the errors in the binary decompostion of the integers. 

### Optimization parameters

The optimization lays on Hamiltonian mechanics theorems that lead to a solving using a symplectic Euler scheme. The latter requires some parameters that you can edit at will and whose functionnality is explained hereafter. To use your personal parameters, simply write:

```
>>> markowitz.optimize(name_of_parameter = value)
```

#### Quantum parameters

The parameters called `detuning_frequency` and `kerr_constant` come straightly from the quantum theory. Their default value were established through scientific studies and it is advised not to change them.

However, you can modify at will the `pressure` parameter as long as you respect some constraints:
- it must be a `lambda` function 
- it must be greater than the `detuning_frequency` at some point, else the scheme will never bifurcate
- keep in mind that the pressure models a slow and steady evolution of the quantum system so it must evolve slowly and continuously

These constraints are actually simple recommendations but not respecting them could (and should) lead to unaccurate and irrelevant results.

#### Euler scheme parameters

The `time_step` represents the disretized time between to time steps in the Euler scheme. Small time steps lead to a higher accuracy but also to longer computation times.

The `symplectic_parameter` represents the number of symplectic loops at each step of the Euler scheme. Its value must be an integer and should be comprised between 2 and 5.

#### Stop criterion

The Euler scheme stops when a stop criterion is satisfied. The latter depends on the `sampling_period` and `display_time` parameters that you can change at will. The higher they are, the more accurate the final result should be, but so will be the computation time. It is recommended to try with different values to test a good batch of parameters.

## Broader use of the algorithm

Even though this package was designed particularly to optimize Markowitz portfolios, it can be used to solve any Ising problem as long as you provide the correlation matrix $J$ and the magnetic field $h$ as follow:

```
>>> from simulated_bifurcation import Ising
>>> ising = Ising(J, h)
```

This model can then be optimized following the same process as for the Markowitz problem.

```
>>> ising.get_ground_state()
```

> The parameters used in the `get_ground_state()` method of the `Ising` class are strictly the same as for the `Markowitz`'s `optimize` method.

To access, the ground state and the energy of the Ising model, simply use:

```
>>> print(ising.ground_state)
```
```
>>> print(ising.energy())
```
