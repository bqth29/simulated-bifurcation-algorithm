from typing import Tuple

import numpy as np
import torch
from sympy import Poly


class ExpressionCompiler:
    """
    Helper class to convert Sympy quadratic multivariate polynomial
    expressions to tensors that respectively represent the constant,
    linear and quadratic coefficients of said polynomial.

    The tensors generated are understandable by SB's native polynomials
    objects.

    The aim of this class is to define polynomials in a more natural way,
    using mathematical equations that are more accessible and less
    cumbersome than tensors, especially for small instances.

    Notes
    -----
    This class only handles order 2 polynomials.

    See Also
    --------
    BaseMultivariateQuadraticPolynomial
    """

    def __init__(self, polynomial: Poly) -> None:
        """
        Parameters
        ----------
        expression : Sympy
            the natural mathematical writing of the polynomial.
        """
        ExpressionCompiler.__check_polynomial_degree(polynomial)
        self.polynomial = polynomial
        self.variables = len(self.polynomial.gens)
        self.__quadratic_coefficients = torch.zeros(self.variables, self.variables)
        self.__linear_coefficients = torch.zeros(self.variables)
        self.__constant_coefficient = torch.tensor([0.0])

    @staticmethod
    def __check_polynomial_degree(polynomial: Poly):
        degree = polynomial.total_degree()
        if degree != 2:
            raise ValueError(f"Expected degree 2 polynomial, got {degree}.")

    def __add_quadratic_coefficient(self, coefficient: float, degrees: Tuple[int]):
        indeces = np.nonzero(degrees)[0]
        if len(indeces) == 1:
            index = indeces[0]
            self.__quadratic_coefficients[index, index] = coefficient
        else:
            index_1, index_2 = indeces[0], indeces[1]
            self.__quadratic_coefficients[index_1, index_2] = coefficient

    def __add_linear_coefficient(self, coefficient: float, degrees: Tuple[int]):
        index = np.nonzero(degrees)[0][0]
        self.__linear_coefficients[index] = coefficient

    def __add_constant_coefficient(self, coefficient: float):
        torch.add(
            self.__constant_coefficient,
            torch.tensor([coefficient]),
            out=self.__constant_coefficient,
        )

    def __fill_tensors(self):
        for coefficient, degrees in zip(
            self.polynomial.coeffs(), self.polynomial.monoms()
        ):
            coefficient = float(coefficient)
            monomial_degree = np.sum(degrees)
            if monomial_degree == 0:
                self.__add_constant_coefficient(coefficient)
            elif monomial_degree == 1:
                self.__add_linear_coefficient(coefficient, degrees)
            else:
                self.__add_quadratic_coefficient(coefficient, degrees)

    def compile(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the coefficient tensors associated to the polynomial
        expression.

        Returns
        -------
        constant : Tensor
            the constant (order 0) coefficient of the polynomial.
        constant : Tensor
            the linear (order 1) coefficients of the polynomial.
            The tensor is one-dimensional.
        quadratic: Tensor
            the quadratic (order 2) coefficients of the polynomial.
            The tensor is two-dimensional and square.
        """
        self.__fill_tensors()
        return (
            self.__constant_coefficient,
            self.__linear_coefficients,
            self.__quadratic_coefficients,
        )
