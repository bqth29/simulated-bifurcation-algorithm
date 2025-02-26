"""
Implementation of utility models to help define and solve optimization
problems with the Simulated Bifurcation algorithm.

Available classes
-----------------
Ising:
    Interface to the Simulated Bifurcation algorithm used for optimizing
    user-defined polynomial.
QuadraticPolynomial:
    Class to implement multivariate quadratic polynomials from SymPy
    polynomial expressions or tensors that can be casted to Ising model
    for Simulated Bifurcation algorithm compatibility purposes.

See Also
--------
simulated_bifurcation:
    Module defining high-level routines for a basic usage.
models:
    Package containing the implementation of several common combinatorial
    optimization problems.

"""

from .ising import Ising
from .quadratic_polynomial import QuadraticPolynomial
