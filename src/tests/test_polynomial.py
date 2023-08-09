import numpy as np
import pytest
import torch

from src.simulated_bifurcation.ising_core import IsingCore
from src.simulated_bifurcation.polynomial import IsingPolynomialInterface

matrix = torch.Tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
vector = torch.Tensor([[1], [2], [3]])
constant = 1


class IsingPolynomialInterfaceImpl(IsingPolynomialInterface):
    def to_ising(self):
        pass  # pragma: no cover

    def convert_spins(self, ising: IsingCore):
        pass  # pragma: no cover


def test_init_polynomial_from_tensors():
    polynomial = IsingPolynomialInterfaceImpl(
        torch.Tensor(matrix), torch.Tensor(vector), constant
    )
    assert torch.equal(polynomial.matrix, torch.Tensor(matrix))
    assert torch.equal(polynomial.vector, torch.Tensor(vector))
    assert polynomial.constant == 1.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.0
    assert torch.equal(polynomial[2], torch.Tensor(matrix))
    assert torch.equal(polynomial[1], torch.Tensor(vector))
    assert polynomial.dtype == torch.float32
    assert polynomial.device == torch.device("cpu")
    with pytest.raises(ValueError):
        # noinspection PyStatementEffect
        polynomial[3]


def test_init_polynomial_from_arrays():
    polynomial = IsingPolynomialInterfaceImpl(
        np.array(matrix), np.array(vector), constant
    )
    assert torch.equal(polynomial.matrix, torch.Tensor(matrix))
    assert torch.equal(polynomial.vector, torch.Tensor(vector))
    assert polynomial.constant == 1.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.0
    assert torch.equal(polynomial[2], torch.Tensor(matrix))
    assert torch.equal(polynomial[1], torch.Tensor(vector))


def test_init_polynomial_without_order_one_and_zero():
    polynomial = IsingPolynomialInterfaceImpl(torch.Tensor(matrix))
    assert torch.equal(polynomial.matrix, torch.Tensor(matrix))
    assert torch.equal(polynomial.vector, torch.zeros(polynomial.dimension))
    assert polynomial.constant == 0.0
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert torch.equal(polynomial[2], torch.Tensor(matrix))
    assert torch.equal(polynomial[1], torch.zeros(polynomial.dimension))
    assert polynomial[0] == 0.0


def test_init_with_wrong_parameters():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        IsingPolynomialInterfaceImpl(None)
    with pytest.raises(ValueError):
        IsingPolynomialInterfaceImpl(torch.unsqueeze(matrix, 0))
    with pytest.raises(ValueError):
        IsingPolynomialInterfaceImpl(
            torch.Tensor(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            )
        )
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        IsingPolynomialInterfaceImpl(matrix, ("hello", "world!"))
    with pytest.raises(ValueError):
        IsingPolynomialInterfaceImpl(matrix, 1)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        IsingPolynomialInterfaceImpl(matrix, constant="hello world!")


def test_call_polynomial():
    polynomial = IsingPolynomialInterfaceImpl(matrix)
    assert polynomial(torch.Tensor([0, 0, 0])) == 0
    assert torch.equal(
        polynomial(torch.Tensor([[0, 0, 0], [1, 2, 3]])), torch.Tensor([0, 228])
    )
    assert polynomial(torch.zeros((1, 5, 3, 1, 2, 1, 3))).shape == (1, 5, 3, 1, 2, 1)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        polynomial("hello world!")
    with pytest.raises(ValueError):
        polynomial(torch.Tensor([1, 2, 3, 4, 5]))


def test_call_polynomial_with_accepted_values():
    polynomial = IsingPolynomialInterfaceImpl(matrix, accepted_values=[0, 1])
    assert polynomial(torch.Tensor([0, 0, 0])) == 0
    with pytest.raises(ValueError):
        polynomial(torch.Tensor([0, 1, 2]))


def test_ising_interface():
    with pytest.raises(NotImplementedError):
        # noinspection PyTypeChecker
        IsingPolynomialInterface.to_ising(None)
    with pytest.raises(NotImplementedError):
        # noinspection PyTypeChecker
        IsingPolynomialInterface.convert_spins(None, None)
