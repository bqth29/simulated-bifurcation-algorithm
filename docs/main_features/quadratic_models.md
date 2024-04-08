# Quadratic Models

The package allows you to directly describe your polynomial instances in the optimization functions, but you can also independently create your model and optimize it afterwards.

## Create a model

### Using tensors

### Using a SymPy expression

[SymPy](https://www.sympy.org/), a symbolic mathematics library for Python, empowers users to express quadratic optimization problems in a natural and concise mathematical form, akin to how they would be written on paper.

> Decision variables in the same order as the expression

### Model dtype

- API dtype (not used for computations)
- Returned values dtype

```{eval-rst}
.. autofunction:: simulated_bifurcation.build_model
```

## Evaluation

## Model optimization

- cast to float
- same parameters 
