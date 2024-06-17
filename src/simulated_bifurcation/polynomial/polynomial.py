from typing import Optional, Sequence, Union

import numpy as np
import torch
from sympy import Poly

from .polynomial_map import PolynomialMap, TensorLike

PolynomialLike = Union[Sequence[TensorLike], Poly]


class Polynomial:
    """
    Class to define multivariate polynomials of any dimension,
    alongside basic polynomial operation.

    Parameters
    ----------
    polynomial : PolynomialLike
        Source data of the multivariate quadratic polynomial to optimize. It can
        be a SymPy polynomial expression or tensors/arrays of coefficients.
        If tensors/arrays are provided, the monomial degree associated to
        the coefficients is the number of dimensions of the tensor/array,
        and all dimensions must be equal. The quadratic tensor must be square
        and symmetric and is mandatory. The linear tensor must be 1-dimensional
        and the constant term can either be a float/int or a 0-dimensional tensor.
        Both are optional. Tensors can be passed in an arbitrary order.

    Keyword-Only Parameters
    -----------------------
    dtype : torch.dtype, default=torch.float32
        Data-type used for the polynomial data.
    device : str | torch.device, default="cpu"
        Device on which the polynomial data is defined.

    Examples
    --------

    Define a degree 7 polynomial from a SymPy expression

      >>> x, y = sympy.symbols("x y")
      >>> expression = sympy.poly(x**7 + 2 * x * y + 1)
      >>> polynomial = Polynomial(expression)
      >>> polynomial.degree
      7
      >>> polynomial.n_variables
      2

    Define a degree 2 polynomial from a tensors

      >>> Q = torch.tensor([[1, -2],
      ...                   [0, 3]])
      >>> l = torch.tensor([5, -2]) # 1-dimensional tensor!
      >>> c = 3
      >>> polynomial = Polynomial(Q, l, c)
      >>> polynomial.degree
      2
      >>> polynomial.n_variables
      2

    """

    def __init__(
        self,
        *_input: PolynomialLike,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if len(_input) == 1 and isinstance(_input[0], Poly):
            self.__polynomial_map = PolynomialMap.from_expression(
                _input[0], dtype=dtype, device=device
            )
        else:
            self.__polynomial_map = PolynomialMap.from_tensors(
                *_input, dtype=dtype, device=device
            )

    @property
    def degree(self) -> int:
        """
        Degree of the polynomial.

        Returns
        -------
        int
        """
        return np.max(list(self.__polynomial_map.keys()))

    @property
    def n_variables(self) -> int:
        """
        Number of variables of the polynomial.

        Returns
        -------
        int

        """
        return self.__polynomial_map.size

    @property
    def device(self) -> torch.device:
        """
        Device on which of all the tensors of the polynomial are defined.

        Returns
        -------
        torch.device

        """
        return self.__polynomial_map.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Common data-type of all the tensors of the polynomial.

        Returns
        -------
        torch.dtype

        """
        return self.__polynomial_map.dtype

    def __getitem__(self, degree: int) -> torch.Tensor:
        if degree in self.__polynomial_map.keys():
            return self.__polynomial_map[degree]
        if isinstance(degree, int):
            if degree >= 0:
                return torch.zeros(
                    (self.n_variables,) * degree, dtype=self.dtype, device=self.device
                )
        raise ValueError("Positive integer required.")

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Method to migrate the tensors that define the polynomial
        to another data-type and/or device. The object is mofified
        in place.

        Parameters
        ----------
        dtype : torch.dtype, optional
            The new data type, by default None.
        device : str | torch.device, optional
            The new device, by default None.

        """
        self.__polynomial_map = PolynomialMap(
            {
                key: value.to(dtype=dtype, device=device)
                for key, value in self.__polynomial_map.items()
            }
        )
