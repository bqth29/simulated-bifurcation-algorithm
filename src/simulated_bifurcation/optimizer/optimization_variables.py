import os
from enum import Enum
from typing import Dict, Optional


def get_env() -> Dict[str, float]:
    return {variable.name: variable.get() for variable in OptimizationVariable}


def set_env(
    time_step: Optional[float] = None,
    pressure_slope: Optional[float] = None,
    heat_coefficient: Optional[float] = None,
):
    if (
        (time_step is None or isinstance(time_step, float))
        and (pressure_slope is None or isinstance(pressure_slope, float))
        and (heat_coefficient is None or isinstance(heat_coefficient, float))
    ):
        OptimizationVariable.TIME_STEP.set(time_step)
        OptimizationVariable.PRESSURE_SLOPE.set(pressure_slope)
        OptimizationVariable.HEAT_COEFFICIENT.set(heat_coefficient)
        return
    raise TypeError(f"All optimization variables must be floats.")


def reset_env() -> None:
    for variable in OptimizationVariable:
        variable.reset()


class OptimizationVariable(Enum):
    TIME_STEP = 0.1, "PYTHON_SB_TIME_STEP"
    PRESSURE_SLOPE = 0.01, "PYTHON_SB_PRESSURE_SLOPE"
    HEAT_COEFFICIENT = 0.06, "PYTHON_SB_HEAT_COEFFICIENT"

    def __init__(self, default_value: float, env_name: str) -> None:
        super().__init__()
        self.__default_value = default_value
        self.__name = env_name[10:].replace("_", " ").lower()
        self.__env_name = env_name
        self.reset()

    def set(self, value: Optional[float]) -> None:
        if value is None:
            return
        if not isinstance(value, float):
            raise TypeError(f"Expected a float but got a {type(value)}.")
        os.environ[self.__env_name] = str(value)
        print(f"Simulated Bifurcation optimizer's {self.__name} set to {value}.")

    def get(self) -> float:
        return float(os.environ[self.__env_name])

    def reset(self) -> None:
        os.environ[self.__env_name] = str(self.__default_value)
