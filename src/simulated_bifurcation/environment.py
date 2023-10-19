import os
from typing import Dict, Optional, Union

import torch

from .optimizer.optimization_variables import OptimizationVariable


def get_env() -> Dict[str, Union[float, torch.dtype, torch.device]]:
    return {
        **{variable.name: variable.get() for variable in OptimizationVariable},
        "DEFAULT_DTYPE": torch.__getattribute__(
            os.environ["PYTHON_SB_DEFAULT_DTYPE"].split(".")[1]
        ),
        "DEFAULT_DEVICE": torch.device(os.environ["PYTHON_SB_DEFAULT_DEVICE"]),
    }


def set_env(
    *,
    time_step: Optional[float] = None,
    pressure_slope: Optional[float] = None,
    heat_coefficient: Optional[float] = None,
    default_dtype: Optional[torch.dtype] = None,
    default_device: Optional[str] = None
):
    __set_optimization_variables(time_step, pressure_slope, heat_coefficient)
    __set_default_dtype(default_dtype)
    __set_default_device(default_device)


def __set_default_device(default_device: Optional[str]):
    if default_device is not None:
        if isinstance(default_device, str):
            try:
                device = torch.device(default_device)
                torch.set_default_device(device)
                os.environ["PYTHON_SB_DEFAULT_DEVICE"] = str(device)
            except:
                raise ValueError("Invalid device type.")
        else:
            raise TypeError("Default device must be a string.")


def __set_default_dtype(default_dtype: Optional[torch.dtype]):
    if default_dtype is not None:
        if isinstance(default_dtype, torch.dtype):
            torch.set_default_dtype(default_dtype)
            os.environ["PYTHON_SB_DEFAULT_DTYPE"] = str(default_dtype)
        else:
            raise TypeError("Default dtype must be a valid torch dtype.")


def __set_optimization_variables(
    time_step: Optional[float],
    pressure_slope: Optional[float],
    heat_coefficient: Optional[float],
):
    if (
        (time_step is None or isinstance(time_step, float))
        and (pressure_slope is None or isinstance(pressure_slope, float))
        and (heat_coefficient is None or isinstance(heat_coefficient, float))
    ):
        OptimizationVariable.TIME_STEP.set(time_step)
        OptimizationVariable.PRESSURE_SLOPE.set(pressure_slope)
        OptimizationVariable.HEAT_COEFFICIENT.set(heat_coefficient)
    else:
        raise TypeError("All optimization variables must be floats.")


def reset_env() -> None:
    for variable in OptimizationVariable:
        variable.reset()
    __set_default_dtype(torch.float32)
    __set_default_device("cpu")
