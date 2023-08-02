import numpy as np
import pytest
import torch

from simulated_bifurcation.ising_core import IsingCore
from src.simulated_bifurcation.polynomial import IsingPolynomialInterface

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
vector = [[1], [2], [3]]
constant = 1


class IsingPolynomialInterfaceImpl(IsingPolynomialInterface):
    def to_ising(self):
        pass

    def convert_spins(self, ising: IsingCore):
        pass


def test_init_polynomial_from_tensors():
    polynomial = IsingPolynomialInterfaceImpl(
        torch.Tensor(matrix), torch.Tensor(vector), constant
    )
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == torch.Tensor(vector))
    assert polynomial.constant == 1.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.0
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == torch.Tensor(vector))
    assert polynomial.dtype == torch.float32
    assert polynomial.device == torch.device("cpu")
    with pytest.raises(ValueError):
        polynomial[3]


def test_init_polynomial_from_arrays():
    polynomial = IsingPolynomialInterfaceImpl(
        np.array(matrix), np.array(vector), constant
    )
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == torch.Tensor(vector))
    assert polynomial.constant == 1.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.0
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == torch.Tensor(vector))


def test_init_polynomial_from_lists():
    polynomial = IsingPolynomialInterfaceImpl(matrix, vector, constant)
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == torch.Tensor(vector))
    assert polynomial.constant == 1.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.0
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == torch.Tensor(vector))


def test_init_polynomial_without_order_one_and_zero():
    polynomial = IsingPolynomialInterfaceImpl(torch.Tensor(matrix))
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == 0)
    assert polynomial.constant == 0.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 0.0
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == 0)


def test_init_with_wrong_parameters():
    with pytest.raises(TypeError):
        IsingPolynomialInterfaceImpl(None)
    with pytest.raises(ValueError):
        IsingPolynomialInterfaceImpl([matrix])
    with pytest.raises(ValueError):
        IsingPolynomialInterfaceImpl(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
    with pytest.raises(TypeError):
        IsingPolynomialInterfaceImpl(matrix, ("hello", "world!"))
    with pytest.raises(ValueError):
        IsingPolynomialInterfaceImpl(matrix, 1)
    with pytest.raises(TypeError):
        IsingPolynomialInterfaceImpl(matrix, constant="hello world!")


def test_call_polynomial():
    polynomial = IsingPolynomialInterfaceImpl(matrix)
    assert polynomial([0, 0, 0]) == 0
    assert polynomial([[0, 1], [0, 2], [0, 3]]) == [0, 228]
    with pytest.raises(TypeError):
        polynomial("hello world!")
    with pytest.raises(ValueError):
        polynomial([1, 2, 3, 4, 5])


def test_call_polynomial_with_accepted_values():
    polynomial = IsingPolynomialInterfaceImpl(matrix, accepted_values=[0, 1])
    assert polynomial([0, 0, 0]) == 0
    with pytest.raises(ValueError):
        polynomial([0, 1, 2])


def test_ising_interface():
    with pytest.raises(NotImplementedError):
        IsingPolynomialInterface.to_ising(None)
    with pytest.raises(NotImplementedError):
        IsingPolynomialInterface.convert_spins(None, None)
