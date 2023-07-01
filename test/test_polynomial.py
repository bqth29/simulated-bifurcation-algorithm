import torch
import pytest
import numpy as np
from src.simulated_bifurcation.polynomial import Polynomial


matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
vector = [[1], [2], [3]]
constant = 1

def test_init_polynomial_from_tensors():
    polynomial = Polynomial(torch.Tensor(matrix), torch.Tensor(vector), constant)
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == torch.Tensor(vector))
    assert polynomial.constant == 1.
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == torch.Tensor(vector))
    
def test_init_polynomial_from_arrays():
    polynomial = Polynomial(np.array(matrix), np.array(vector), constant)
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == torch.Tensor(vector))
    assert polynomial.constant == 1.
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == torch.Tensor(vector))

def test_init_polynomial_from_lists():
    polynomial = Polynomial(matrix, vector, constant)
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == torch.Tensor(vector))
    assert polynomial.constant == 1.
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 1.
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == torch.Tensor(vector))

def test_init_polynomial_without_order_one_and_zero():
    polynomial = Polynomial(torch.Tensor(matrix))
    assert torch.all(polynomial.matrix == torch.Tensor(matrix))
    assert torch.all(polynomial.vector == 0)
    assert polynomial.constant == 0.
    assert polynomial.dimension == 3
    assert len(polynomial) == 3
    assert polynomial[0] == 0.
    assert torch.all(polynomial[2] == torch.Tensor(matrix))
    assert torch.all(polynomial[1] == 0)

def test_init_with_wrong_parameters():
    with pytest.raises(TypeError):
        Polynomial(None)
    with pytest.raises(ValueError):
        Polynomial([matrix])
    with pytest.raises(ValueError):
        Polynomial([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(TypeError):
        Polynomial(matrix, ('hello', 'world!'))
    with pytest.raises(ValueError):
        Polynomial(matrix, 1)
    with pytest.raises(TypeError):
        Polynomial(matrix, constant='hello world!')

def test_call_polynomial():
    polynomial = Polynomial(matrix)
    assert polynomial([0, 0, 0]) == 0
    assert polynomial([[0, 1], [0, 2], [0, 3]]) == [0, 228]
    with pytest.raises(TypeError):
        polynomial('hello world!')
    with pytest.raises(ValueError):
        polynomial([1, 2, 3, 4, 5])

def test_call_polynomial_with_accepted_values():
    polynomial = Polynomial(matrix, accepted_values=[0, 1])
    assert polynomial([0, 0, 0]) == 0
    with pytest.raises(ValueError):
        polynomial([0, 1, 2])
