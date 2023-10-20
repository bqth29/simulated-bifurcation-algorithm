"""
Implementation of multivariate degree 2 polynomials.

.. deprecated:: 1.2.1
    `IsingPolynomialInterface` will be removed in simulated-bifurcation
    1.3.0, it is replaced by `BaseMultivariateQuadraticPolynomial` in
    prevision of the addition of multivariate polynomials of an arbitrary
    degree.
    `BinaryPolynomial` will be modified in simulated-bifurcation 1.3.0, it
    is replaced by `BinaryQuadraticPolynomial` in prevision of the addition
    of multivariate polynomials of an arbitrary degree.
    `IntegerPolynomial` will be modified in simulated-bifurcation 1.3.0, it
    is replaced by `IntegerQuadraticPolynomial` in prevision of the
    addition of multivariate polynomials of an arbitrary degree.
    `SpinPolynomial` will be modified in simulated-bifurcation 1.3.0, it is
    replaced by `SpinQuadraticPolynomial` in prevision of the addition of
    multivariate polynomials of an arbitrary degree.

Multivariate degree 2 polynomials are the sum of a quadratic form and a
linear form plus a constant term:
`ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
or `x.T Q x + l.T x + c` in matrix notation,
where `Q` is a square matrix, `l` a vector a `c` a constant.
The set to which the vector `x` is called the domain of the polynomial, and
the polynomial is said to be defined over a domain.

Available classes
-----------------
BaseMultivariateQuadraticPolynomial:
    Abstract class for multivariate degree 2 polynomials.
BinaryQuadraticPolynomial:
    Multivariate degree 2 polynomials over vectors whose entries are in
    {0, 1}.
IntegerQuadraticPolynomial:
    Multivariate degree 2 polynomials over non-negative integers with a
    fixed number of bits. For instance, a polynomial over 7-bits integers
    is a polynomial whose domain is the set of vectors whose entries are
    all 7-bits integers, that is integer between 1 and 2^7 - 1 = 127
    (inclusive).
SpinQuadraticPolynomial:
    Multivariate degree 2 polynomials over vectors whose entries are in
    {-1, 1}.

See Also
--------
simulated_bifurcation:
    Module defining high-level routines for a basic usage.
models:
    Package containing the implementation of several common combinatorial
    optimization problems.

"""


from .base_multivariate_polynomial import (
    BaseMultivariateQuadraticPolynomial,
    IsingPolynomialInterface,
)
from .binary_polynomial import BinaryPolynomial, BinaryQuadraticPolynomial
from .integer_polynomial import IntegerPolynomial, IntegerQuadraticPolynomial
from .spin_polynomial import SpinPolynomial, SpinQuadraticPolynomial
