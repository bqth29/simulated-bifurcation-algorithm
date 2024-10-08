# Quadratic Models

The package allows you to directly describe your polynomial instances in the optimization functions, but you can also independently create your model and optimize it afterwards.

## Create a model

### Using tensors

<!-- TODO: examples with arbitrary order -->

### Using a SymPy Poly

[SymPy](https://www.sympy.org/), a symbolic mathematics library for Python, empowers users to express quadratic optimization problems in a natural and concise mathematical form, akin to how they would be written on paper.

<!--
    TODO:
    - variables in the created model have the same order as the generators of the Poly
-->

> Decision variables in the same order as the expression

### Model dtype

- API dtype (not used for computations)
- Returned values dtype

### Model device

## Evaluation

<!--
    TODO:
    - example
-->

## Model optimization

<!--
    TODO:
    - example
    - explain the difference between model dtype and optimization dtype
-->

```{eval-rst}
.. autofunction:: simulated_bifurcation.build_model
```
