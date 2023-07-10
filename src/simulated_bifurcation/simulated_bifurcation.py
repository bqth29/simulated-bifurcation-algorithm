from .polynomial import IsingInterface, SpinPolynomial, BinaryPolynomial, IntegerPolynomial
from numpy import ndarray
from typing import Tuple, Union
from torch import Tensor, dtype, float32
from re import compile


def minimize(matrix: Union[Tensor, ndarray], vector: Union[Tensor, ndarray] = None, constant: float = None,
    input_type: str = 'spin', dtype: dtype = float32, device: str = 'cpu', convergence_threshold: int = 50,
    sampling_period: int = 50, max_steps: int = 10000, agents: int = 128, use_window: bool = True,
    ballistic: bool = False, heat: bool = False, verbose: bool = True) -> Tuple[Tensor, float]:
    model = _build_model(matrix=matrix, vector=vector, constant=constant, input_type=input_type, dtype=dtype, device=device)
    best_vector = model.optimize(convergence_threshold=convergence_threshold, sampling_period=sampling_period,
        max_steps=max_steps, agents=agents, use_window=use_window, ballistic=ballistic, heat=heat,
        verbose=verbose, minimize=True)
    best_value = model(best_vector)
    return best_vector, best_value

def maximize(matrix: Union[Tensor, ndarray], vector: Union[Tensor, ndarray] = None, constant: float = None,
    input_type: str = 'spin', dtype: dtype = float32, device: str = 'cpu', convergence_threshold: int = 50,
    sampling_period: int = 50, max_steps: int = 10000, agents: int = 128, use_window: bool = True,
    ballistic: bool = False, heat: bool = False, verbose: bool = True) -> Tuple[Tensor, float]:
    model = _build_model(matrix=matrix, vector=vector, constant=constant, input_type=input_type, dtype=dtype, device=device)
    best_vector = model.optimize(convergence_threshold=convergence_threshold, sampling_period=sampling_period,
        max_steps=max_steps, agents=agents, use_window=use_window, ballistic=ballistic, heat=heat,
        verbose=verbose, minimize=False)
    best_value = model(best_vector)
    return best_vector, best_value

def _build_model(matrix: Union[Tensor, ndarray], vector: Union[Tensor, ndarray], constant: float, input_type: str, dtype: dtype, device: str) -> IsingInterface:
    int_type_regex = compile(r'int(\d+)')
    if input_type == 'spin':
        return SpinPolynomial(matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device)
    if input_type == 'binary':
        return BinaryPolynomial(matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device)
    if int_type_regex.match(input_type):
        number_of_bits = int(int_type_regex.findall(input_type)[0])
        return IntegerPolynomial(matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device, number_of_bits=number_of_bits)
    raise ValueError(r'Input type must match spin, binary or int\d+.')
