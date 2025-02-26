# Quadratic Models

The package allows you to directly describe your polynomial instances in the optimization functions, but you can also independently create your model and optimize it afterwards.

## Create a model

In this section, will consider the following binary optimization problem:

$$\text{Maximize } 8x^{2} - 17xy + 4y^{2} + 11x -2y + 5$$

with $x$ and $y$ being either $0$ or $1$, and see how to model and solve it using the Simulated Bifurcation algorithm.

### Using tensors

Quadratic problems can be written in matrix form with a square matrix that bears the quadratic coefficients, a vector that bears the linear coefficient and a scalar term that works as an offset. For the aformentioned, the quadratic matrix would be $\begin{bmatrix} 8 & -17 \\ 4 & 0 \end{bmatrix}$, the linear vector would be $\begin{bmatrix} 11 \\ -2 \end{bmatrix}$ and the offset would be $5$.

As this matrix form is often convenient, this implementation of the Simulated Bifurcation algorithm works with PyTorch coefficient tensors for optimization computations.

```python
import simulated_bifurcation as sb


# Define the tensors
Q = torch.tensor([[8, -17], [4, 0]])
l = torch.tesor([11, -2])
c = torch.tensor(5)  # c = 5 would also work

# Solve using the Simulated Bifurcation algorithm
sb.maximize(Q, l, c, domain="spin") 
```

> The order in which the tensors are passed is arbitrary as long as one of each degree is provided at most, except for the quadratic tensor which is mandatory. Thus, for problems that only contain quadratic coefficients, only the quadratic tensor is necessary.

### Using a SymPy Poly

[SymPy](https://www.sympy.org/), a symbolic mathematics library for Python, empowers users to express quadratic optimization problems in a natural and concise mathematical form, akin to how they would be written on paper. This implementation of the Simulated Bifurcation algorithm has been designed to work with SymPy polynomials for a better readability and a simplified user experience.

```python
import sympy
import simulated_bifurcation as sb


# Define the variables
x, y = sympy.symbols("x y")

# Create the polynomial
p = sympy.poly(8 * x ** 2 - 17 * x * y + 4 * y ** 2 + 11 * x - 2 * y + 5)

# Solve using the Simulated Bifurcation algorithm
sb.maximize(p, domain="spin") 
```

> The order of the decision variables in the polynomial is the same as in the expression.

### Model dtype

A quadratic model is given a PyTorch's dtype that defines the type of its coefficients. When the polynomial is evaluated, the output values are typed using the same dtype.

> ⚠️ The model's dtype is not used for optimization computations. A specific parameter called `dtype` serves this purpose in the optimization methods. For more details, see the [page](simulated_bifurcation_optimizer.md) dedicated to parameters.

### Model device

A quadratic model can also be defined on a specific device. By default, its tensors are stored on the CPU but they can also be defined to work on a GPU if the hardware is PyTorch-compatible.

## Build a model

```{eval-rst}
.. autofunction:: simulated_bifurcation.build_model
```

<!--
    TODO:
    ## Evaluation
    ## Model optimization
-->
