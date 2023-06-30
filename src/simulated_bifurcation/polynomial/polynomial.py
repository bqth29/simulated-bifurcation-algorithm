from ..ising import Ising
from abc import ABC, abstractmethod
from typing import Any, Iterable, final, List, Union
import torch
import numpy as np


class Polynomial():

    """
    Order two multivariate polynomial. Sum of a
    quadratic form, a linear form and a constant
    term.
    """

    def __init__(self, matrix: Union[torch.Tensor, np.ndarray], vector: Union[torch.Tensor, np.ndarray, None],
                 constant: Union[int, float, None], accepted_values: Union[None, List[int]],
                 dtype: torch.dtype, device: str) -> None:
        self.__init_matrix(matrix, dtype, device)
        self.__init_vector(vector, dtype, device)
        self.__init_constant(constant)
        self.__accepted_values = accepted_values[:]

    @property
    def matrix(self) -> torch.Tensor:
        return self.__matrix
    
    @property
    def vector(self) -> torch.Tensor:
        return self.__vector.reshape(-1, 1)
    
    @property
    def constant(self) -> float:
        return float(self.__constant)
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    
    @property
    def dtype(self) -> torch.dtype:
        return self.__matrix.dtype
    
    @property
    def device(self) -> torch.device:
        return self.__matrix.device

    @final
    def __call__(self, value: Union[torch.Tensor, np.ndarray, List[List[float]]]) -> Union[float, List[float]]:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.Tensor(value)
            except:
                raise TypeError(f"Input value cannot be cast to Tensor.")
        if (self.__accepted_values is not None) and (not np.all(np.isin(value.numpy(), self.__accepted_values))):
            raise ValueError(f'Input values must all belong to {self.__accepted_values}.')
        if value.shape in [(self.dimension,), (self.dimension, 1), (1, self.dimension)]:
            value = value.reshape((-1, 1))
            value = value.t() @ self.matrix @ value + value.t() @ self.vector + self.constant
            return value.item()
        if value.shape[0] == self.dimension:
            values = torch.einsum('ij, ji -> i', value.t(), self.matrix @ value + self.vector) + self.constant
            return values.tolist()
        raise ValueError(f"Expected {self.dimension} rows, got {value.shape[0]}.")
    
    @final
    def __getitem__(self, coefficient: int) -> Union[torch.Tensor, float]:
        if coefficient == 0:
            return self.constant
        if coefficient == 1:
            return self.vector
        if coefficient == 2:
            return self.matrix
        raise ValueError('Only accepts 0, 1 or 2 as arguments.')
    
    @final
    def __len__(self) -> int:
        return self.__dimension
    
    def __init_matrix(self, matrix: Iterable, dtype: torch.dtype, device: str) -> None:
        tensor_matrix = self.__cast_matrix_to_tensor(matrix, dtype, device)
        self.__check_square_matrix(tensor_matrix)
        self.__matrix = tensor_matrix
        self.__dimension = tensor_matrix.shape[0]

    def __init_vector(self, vector: Iterable, dtype: torch.dtype, device: str) -> None:
        tensor_vector = self.__cast_vector_to_tensor(vector, dtype, device)
        self.__check_vector_shape(tensor_vector)
        self.__vector = tensor_vector

    def __init_constant(self, constant: Union[float, int, None]) -> None:
        self.__constant = self.__cast_constant_to_float(constant)
    
    def __cast_matrix_to_tensor(self, matrix: Iterable, dtype: torch.dtype, device: str) -> torch.Tensor:
        try:
            return torch.Tensor(matrix).to(device=device, dtype=dtype)
        except:
            raise TypeError('Matrix cannot be cast to tensor.')
    
    def __check_square_matrix(self, matrix: torch.Tensor) -> None:
        if len(matrix.shape) != 2:
            raise ValueError(f'Matrix requires two dimension, got {len(matrix.shape)}.')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Matrix must be square.')
        
    def __cast_vector_to_tensor(self, vector: Union[Iterable, None], dtype: torch.dtype, device: str) -> torch.Tensor:
        if vector is None:
            return torch.zeros(self.dimension, dtype=dtype, device=device)
        try:
            return torch.Tensor(vector).to(device=device, dtype=dtype)
        except:
            raise TypeError('Vector cannot be cast to tensor.')
        
    def __check_vector_shape(self, vector: torch.Tensor) -> None:
        allowed_shapes = [(self.dimension,), (self.dimension, 1), (1, self.dimension)]
        if vector.shape not in allowed_shapes:
            raise ValueError(f'Vector must be of size {self.dimension}, got {vector.shape}.')
        
    def __cast_constant_to_float(self, constant: Union[float, int, None]) -> float:
        if constant is None:
            return 0.
        try:
            return float(constant)
        except:
            raise TypeError('Constant cannot be cast to float.')
