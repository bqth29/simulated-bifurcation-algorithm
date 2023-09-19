"""
Implementation of multivariate degree 2 polynomials over binary vectors.

Multivariate degree 2 polynomials are the sum of a quadratic form and a
linear form plus a constant term:
`ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
or `x.T Q x + l.T x + c` in matrix notation,
where `Q` is a square matrix, `l` a vector a `c` a constant.
The `x(i)`'s values must be binary (either `0` or `1`).

Such polynomials can be translated into equivalent Ising problems and
solved with the Simulated Bifurcation algorithm.

See Also
--------
IsingPolynomialInterface:
    Abstract class for multivariate degree 2 polynomials.
IntegerPolynomial:
    Multivariate degree 2 polynomials over non-negative integers with a
    fixed number of bits.
SpinPolynomial:
    Multivariate degree 2 polynomials over vectors whose entries are in
    {-1, 1}.
models.QUBO: Implementation of the QUBO problem.
models:
    Package containing the implementation of several common
    combinatorial optimization problems.

"""

from typing import Optional, Union

import numpy as np
import torch

from ..ising_core import IsingCore
from .ising_polynomial_interface import IsingPolynomialInterface


class BinaryPolynomial(IsingPolynomialInterface):

    """
    Multivariate degree 2 polynomials over binary vectors.

    Multivariate degree 2 polynomials are the sum of a quadratic form and a
    linear form plus a constant term:
    `ΣΣ Q(i,j)x(i)x(j) + Σ l(i)x(i) + c`
    or `x.T Q x + l.T x + c` in matrix notation,
    where `Q` is a square matrix, `l` a vector a `c` a constant.
    The `x(i)`'s values must be binary (either `0` or `1`).

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

    See Also
    --------
    IsingPolynomialInterface:
        Abstract class for multivariate degree 2 polynomials.
    IntegerPolynomial:
        Multivariate degree 2 polynomials over non-negative integers with a
        fixed number of bits.
    SpinPolynomial:
        Multivariate degree 2 polynomials over vectors whose entries are in
        {-1, 1}.
    models.QUBO: Implementation of the QUBO problem.
    models:
        Package containing the implementation of several common
        combinatorial optimization problems.

    """

    def __init__(
        self,
        matrix: Union[torch.Tensor, np.ndarray],
        vector: Union[torch.Tensor, np.ndarray, None] = None,
        constant: Union[float, int, None] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(matrix, vector, constant, [0, 1], dtype, device)

    def to_ising(self) -> IsingCore:
        symmetrical_matrix = IsingCore.symmetrize(self.matrix)
        J = -0.5 * symmetrical_matrix
        h = 0.5 * self.vector + 0.5 * symmetrical_matrix @ torch.ones(
            len(self), dtype=self.dtype, device=self.device
        )
        return IsingCore(J, h, self.dtype, self.device)

    def convert_spins(self, ising: IsingCore) -> Optional[torch.Tensor]:
        if ising.computed_spins is not None:
            binary_vars = (ising.computed_spins + 1) / 2
        else:
            binary_vars = None
        return binary_vars
