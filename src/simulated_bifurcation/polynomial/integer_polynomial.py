"""
Implementation of multivariate degree 2 polynomials over integer vectors.

.. deprecated:: 1.2.1
  `IntegerPolynomial` and `IntegerQuadraticPolynomial`. Achieving a similar
  behaviour will be done by setting `domain="int..."` when creating a
  polynomial (where ... should be replaced by a positive integer
  corresponding to the number of bits of the integers).

Multivariate degree 2 polynomials are the sum of a quadratic form and a
linear form plus a constant term:
`ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
or `x.T Q x + l.T x + c` in matrix notation,
where `Q` is a square matrix, `l` is a vector and `c` is a constant.
The `x(i)`'s values must be non-negative integers.

Such polynomials can be translated into equivalent Ising problems and
solved with the Simulated Bifurcation algorithm (provided the entries of
the input vectors have a fixed bit-width; for instance 7-bits integers,
that is integers between 0 and 127 inclusive).

See Also
--------
BaseMultivariateQuadraticPolynomial:
    Abstract class for multivariate degree 2 polynomials.
BinaryQuadraticPolynomial:
    Multivariate degree 2 polynomials over vectors whose entries are in
    {0, 1}.
SpinQuadraticPolynomial:
    Multivariate degree 2 polynomials over vectors whose entries are in
    {-1, 1}.
models:
    Package containing the implementation of several common
    combinatorial optimization problems.

"""

import warnings
from typing import Optional, Union

import numpy as np
import torch

from ..ising_core import IsingCore
from .base_multivariate_polynomial import BaseMultivariateQuadraticPolynomial


class IntegerQuadraticPolynomial(BaseMultivariateQuadraticPolynomial):

    """
    Multivariate degree 2 polynomials over fixed bit-width integer vectors.

    .. deprecated:: 1.2.1
        `IntegerQuadraticPolynomial` will be removed in
        simulated-bifurcation 1.3.0. Achieving a similar behaviour will be
        done by setting `domain="int..."` when creating a polynomial (where
        ... should be replaced by a positive integer corresponding to the
        number of bits of the integers).

    Multivariate degree 2 polynomials are the sum of a quadratic form and a
    linear form plus a constant term:
    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
    or `x.T Q x + l.T x + c` in matrix notation,
    where `Q` is a square matrix, `l` is a vector and `c` is a constant.
    The `x(i)`'s values  must be non-negative integers with a fixed
    bit-width. For instance 7-bits integers are the integers between 0 and
    127 inclusive.

    This polynomial can be translated into an equivalent Ising problem and
    solved with the Simulated Bifurcation algorithm.

    Parameters
    ----------
    matrix : (M, M) Tensor | ndarray
        Matrix corresponding to the quadratic terms of the polynomial
        (quadratic form). It should be a square matrix, but not necessarily
        symmetric.
    vector : (M,) Tensor | ndarray | None, optional
        Vector corresponding to the linear terms of the polynomial (linear
        form). The default is None which signifies there are no linear
        terms, that is `vector` is the null vector.
    constant : int | float | None, optional
        Constant of the polynomial. The default is None which signifies
        there is no constant term, that is `constant` = 0.
    number_of_bits: int, default=1
        The number of bits of the input vectors. For instance 7-bits
        integers are the integers between 0 and 127 inclusive.
    dtype : torch.dtype, default=torch.float32
        Data-type used for storing the coefficients of the polynomial.
    device : str | torch.device, default="cpu"
        Device on which the polynomial is located. If available, use "cuda"
        to use the polynomial on a GPU.

    Attributes
    ----------
    matrix
    vector
    constant
    dimension
    device
    dtype
    sb_result : (M, A) Tensor | None
        Spin vectors obtained after optimizing the polynomial. None if no
        optimization method has been called.
    number_of_bits : int
        The number of bits of the input vectors. For instance 7-bits
        integers are the integers between 0 and 127 inclusive.

    See Also
    --------
    BinaryQuadraticPolynomial:
        Multivariate degree 2 polynomials over vectors whose entries are in
        {0, 1}.
    BaseMultivariateQuadraticPolynomial:
        Abstract class for multivariate degree 2 polynomials.
    SpinQuadraticPolynomial:
        Multivariate degree 2 polynomials over vectors whose entries are in
        {-1, 1}.
    models:
        Package containing the implementation of several common
        combinatorial optimization problems.

    """

    def __init__(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        vector: Union[torch.Tensor, np.ndarray, None] = None,
        constant: Union[float, int, None] = None,
        number_of_bits: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        *,
        silence_deprecation_warning=False,
    ) -> None:
        if not silence_deprecation_warning:
            # 2023-11-21, 1.2.1
            warnings.warn(
                "`IntegerQuadraticPolynomial` is deprecated as of simulated-bifurcation"
                " 1.2.1, and it will be removed in simulated-bifurcation 1.3.0. "
                "Achieving a similar behaviour will be done by setting "
                '`domain="int..."` when creating a polynomial. (where ... should be '
                "replaced by a positive integer corresponding to the number of bits of "
                "the integers)",
                DeprecationWarning,
                stacklevel=3,
            )
        if not isinstance(number_of_bits, int) or number_of_bits < 1:
            raise ValueError("The number of bits must be a non-negative integer.")
        super().__init__(
            matrix, vector, constant, [*range(2**number_of_bits)], dtype, device
        )
        self.number_of_bits = number_of_bits
        self.__int_to_bin_matrix = self.integer_to_binary_matrix(
            self.dimension, self.number_of_bits, self.device
        )

    @staticmethod
    def integer_to_binary_matrix(
        dimension: int, number_of_bits: int, device: Union[str, torch.device]
    ) -> torch.Tensor:
        matrix = torch.zeros((dimension * number_of_bits, dimension), device=device)
        for row in range(dimension):
            for col in range(number_of_bits):
                matrix[row * number_of_bits + col][row] = 2.0**col
        return matrix

    def to_ising(self) -> IsingCore:
        symmetrical_matrix = 0.5 * (self.matrix + self.matrix.t())
        J = (
            -0.5
            * self.__int_to_bin_matrix
            @ symmetrical_matrix
            @ self.__int_to_bin_matrix.t()
        )
        h = (
            0.5 * self.__int_to_bin_matrix @ self.vector
            + 0.5
            * self.__int_to_bin_matrix
            @ self.matrix
            @ self.__int_to_bin_matrix.t()
            @ torch.ones(
                (self.dimension * self.number_of_bits),
                dtype=self.dtype,
                device=self.device,
            )
        )
        return IsingCore(J, h, self.dtype, self.device)

    def convert_spins(self, ising: IsingCore) -> Optional[torch.Tensor]:
        """
        Convert the spins of an Ising problem into integer variables.

        Parameters
        ----------
        ising : IsingCore
            The Ising problem whose spins are converted into integer
            variables.

        Returns
        -------
        binary_vars : Tensor | None
            The integer variables corresponding to the spins of `ising`, it
            is None if `ising.computed_spins` is None.

        """
        if ising.computed_spins is not None:
            int_vars = 0.5 * self.__int_to_bin_matrix.t() @ (ising.computed_spins + 1)
        else:
            int_vars = None
        return int_vars


class IntegerPolynomial(IntegerQuadraticPolynomial):

    """
    .. deprecated:: 1.2.1
        `IntegerPolynomial` will be modified in simulated-bifurcation
        1.3.0, it is replaced by `IntegerQuadraticPolynomial` in
        prevision of the addition of multivariate polynomials of an
        arbitrary degree.

    """

    def __init__(self, *args, **kwargs) -> None:
        # 2023-10-03, 1.2.1
        warnings.warn(
            "`IntegerPolynomial` is deprecated as of simulated-bifurcation 1.2.1, and "
            "it will be removed in simulated-bifurcation 1.3.0. Achieving a similar "
            'behaviour will be done by setting `domain="int..."` when creating a '
            "polynomial. (where ... should be replaced by a positive integer "
            "corresponding to the number of bits of the integers)",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__init__(*args, **kwargs, silence_deprecation_warning=True)
