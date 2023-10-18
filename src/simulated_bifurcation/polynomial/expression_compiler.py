from typing import Tuple

import numpy as np
import torch
from sympy import Poly


class ExpressionCompiler:
    def __init__(self, polynomial: Poly) -> None:
        ExpressionCompiler.__check_polynomial_degree(polynomial)
        self.polynomial = polynomial
        self.variables = len(self.polynomial.gens)
        self.order_2_tensor = torch.zeros(self.variables, self.variables)
        self.order_1_tensor = torch.zeros(self.variables)
        self.order_0_tensor = torch.tensor([0.0])

    @staticmethod
    def __check_polynomial_degree(polynomial: Poly):
        degree = polynomial.total_degree()
        if degree != 2:
            raise ValueError(f"Expected degree 2 polynomial, got {degree}.")

    def __add_order_2_coefficient(self, coefficient: float, degrees: Tuple[int]):
        indeces = np.nonzero(degrees)[0]
        if len(indeces) == 1:
            index = indeces[0]
            self.order_2_tensor[index, index] = coefficient
        else:
            index_1, index_2 = indeces[0], indeces[1]
            self.order_2_tensor[index_1, index_2] = coefficient

    def __add_order_1_coefficient(self, coefficient: float, degrees: Tuple[int]):
        index = np.nonzero(degrees)[0][0]
        self.order_1_tensor[index] = coefficient

    def __add_order_0_coefficient(self, coefficient: float):
        torch.add(
            self.order_0_tensor, torch.tensor([coefficient]), out=self.order_0_tensor
        )

    def __fill_tensors(self):
        for coefficient, degrees in zip(
            self.polynomial.coeffs(), self.polynomial.monoms()
        ):
            coefficient = float(coefficient)
            monomial_degree = np.sum(degrees)
            if monomial_degree == 0:
                self.__add_order_0_coefficient(coefficient)
            elif monomial_degree == 1:
                self.__add_order_1_coefficient(coefficient, degrees)
            else:
                self.__add_order_2_coefficient(coefficient, degrees)

    def compile(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__fill_tensors()
        return self.order_0_tensor, self.order_1_tensor, self.order_2_tensor
