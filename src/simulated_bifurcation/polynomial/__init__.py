"""
Implementation of multivariate polynomials.

Available classes
-----------------
Polynomial:
    Class to define multivariate polynomials of any dimension,
    alongside basic polynomial operation.
PolynomialMap:
    Utility class to provide a data structure that properly defined
    multivariate polynomials of any dimension.

See Also
--------
simulated_bifurcation:
    Module defining high-level routines for a basic usage.
core:
    Module of utility models to help define and solve optimization
    problems with the Simulated Bifurcation algorithm.
models:
    Package containing the implementation of several common combinatorial
    optimization problems.

"""

from .polynomial import Polynomial, PolynomialLike
from .polynomial_map import PolynomialMap
