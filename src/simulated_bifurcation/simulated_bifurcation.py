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


def optimize(
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
    best_only: bool = True,
    minimize: bool = True,
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    model = build_model(
        matrix=matrix,
        vector=vector,
        constant=constant,
        input_type=input_type,
        dtype=dtype,
        device=device,
    )
    result, evaluation = model.optimize(
        convergence_threshold=convergence_threshold,
        sampling_period=sampling_period,
        max_steps=max_steps,
        agents=agents,
        use_window=use_window,
        ballistic=ballistic,
        heat=heat,
        verbose=verbose,
        minimize=minimize,
        best_only=best_only,
    )
    return result, evaluation


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
    best_only: bool = True,
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    return optimize(
        matrix,
        vector,
        constant,
        input_type,
        dtype,
        device,
        convergence_threshold,
        sampling_period,
        max_steps,
        agents,
        use_window,
        ballistic,
        heat,
        verbose,
        best_only,
        True,
    )


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
    best_only: bool = True,
) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
    return optimize(
        matrix,
        vector,
        constant,
        input_type,
        dtype,
        device,
        convergence_threshold,
        sampling_period,
        max_steps,
        agents,
        use_window,
        ballistic,
        heat,
        verbose,
        best_only,
        False,
    )


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
