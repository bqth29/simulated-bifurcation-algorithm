import re
from typing import Tuple, Union

import torch
from numpy import ndarray

from .polynomial import (
    BinaryPolynomial,
    IntegerPolynomial,
    IsingPolynomialInterface,
    SpinPolynomial,
)


def minimize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray] = None,
    constant: float = None,
    input_type: str = "spin",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    convergence_threshold: int = 50,
    sampling_period: int = 50,
    max_steps: int = 10000,
    agents: int = 128,
    use_window: bool = True,
    ballistic: bool = False,
    heat: bool = False,
    verbose: bool = True,
    best_only=True,
) -> Tuple[torch.Tensor, float]:
    model = build_model(
        matrix=matrix,
        vector=vector,
        constant=constant,
        input_type=input_type,
        dtype=dtype,
        device=device,
    )
    result = model.optimize(
        convergence_threshold=convergence_threshold,
        sampling_period=sampling_period,
        max_steps=max_steps,
        agents=agents,
        use_window=use_window,
        ballistic=ballistic,
        heat=heat,
        verbose=verbose,
        minimize=True,
        best_only=best_only,
    )
    evaluation = model(result)
    return result, evaluation


def maximize(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray] = None,
    constant: float = None,
    input_type: str = "spin",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    convergence_threshold: int = 50,
    sampling_period: int = 50,
    max_steps: int = 10000,
    agents: int = 128,
    use_window: bool = True,
    ballistic: bool = False,
    heat: bool = False,
    verbose: bool = True,
    best_only=True,
) -> Tuple[torch.Tensor, float]:
    model = build_model(
        matrix=matrix,
        vector=vector,
        constant=constant,
        input_type=input_type,
        dtype=dtype,
        device=device,
    )
    result = model.optimize(
        convergence_threshold=convergence_threshold,
        sampling_period=sampling_period,
        max_steps=max_steps,
        agents=agents,
        use_window=use_window,
        ballistic=ballistic,
        heat=heat,
        verbose=verbose,
        minimize=False,
        best_only=best_only,
    )
    evaluation = model(result)
    return result, evaluation


def build_model(
    matrix: Union[torch.Tensor, ndarray],
    vector: Union[torch.Tensor, ndarray],
    constant: float,
    input_type: str,
    dtype: torch.dtype,
    device: str,
) -> IsingPolynomialInterface:
    int_type_regex = re.compile(r"int(\d+)")
    if input_type == "spin":
        return SpinPolynomial(
            matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device
        )
    if input_type == "binary":
        return BinaryPolynomial(
            matrix=matrix, vector=vector, constant=constant, dtype=dtype, device=device
        )
    if int_type_regex.match(input_type):
        number_of_bits = int(int_type_regex.findall(input_type)[0])
        return IntegerPolynomial(
            matrix=matrix,
            vector=vector,
            constant=constant,
            dtype=dtype,
            device=device,
            number_of_bits=number_of_bits,
        )
    raise ValueError(r"Input type must match spin, binary or int\d+.")
