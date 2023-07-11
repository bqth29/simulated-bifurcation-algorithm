import os
from enum import Enum
from typing import Dict, Union


def get_env() -> Dict[str, float]:
    return {
        variable.name: variable.get()
        for variable in OptimizationVariable
    }

def set_env(time_step: float = None, pressure_slope: float = None, heat_coefficient: float = None):
    if (time_step is None or isinstance(time_step, float)) \
        and (pressure_slope is None or isinstance(pressure_slope, float)) \
        and (heat_coefficient is None or isinstance(heat_coefficient, float)):
        OptimizationVariable.TIME_STEP.set(time_step)
        OptimizationVariable.PRESSURE_SLOPE.set(pressure_slope)
        OptimizationVariable.HEAT_COEFFICIENT.set(heat_coefficient)
        return
    raise TypeError(f'All optimization variables must be floats.')

def reset_env() -> None:
    for variable in OptimizationVariable:
        variable.reset()


class OptimizationVariable(Enum):

    TIME_STEP = .1, 'time step', 'PYTHON_SB_TIME_STEP'
    PRESSURE_SLOPE = .01, 'pressure slope','PYTHON_SB_PRESSURE_SLOPE'
    HEAT_COEFFICIENT = .06, 'heat coefficient', 'PYTHON_SB_HEAT_COEFFICIENT'

    def __init__(self, default_value: float, name: str, env_name: str) -> None:
        super().__init__()
        self.__default_value = default_value
        self.__name = name
        self.__env_name = env_name
        self.reset()

    def set(self, value: Union[float, None]) -> None:
        if value is None:
            return
        if not isinstance(value, float):
            raise TypeError(f'Expected a float but got a {type(value)}.')
        os.environ[self.__env_name] = str(value)
        print(f"Simulated Bifurcation optimizer's {self.__name} set to {value}.")

    def get(self) -> float:
        return float(os.environ[self.__env_name])
    
    def reset(self) -> None:
        os.environ[self.__env_name] = str(self.__default_value)
