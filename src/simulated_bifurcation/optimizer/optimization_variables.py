import os
from enum import Enum
from typing import Optional

ENV_PREFIX = "PYTHON_SB_"


class OptimizationVariable(Enum):
    TIME_STEP = 0.1
    PRESSURE_SLOPE = 0.01
    HEAT_COEFFICIENT = 0.06

    def __init__(self, default_value: float) -> None:
        super().__init__()
        self.__default_value = default_value
        self.__env_name = ENV_PREFIX + self.name
        self.reset()

    def set(self, value: Optional[float]) -> None:
        if value is None:
            return
        if not isinstance(value, float):
            raise TypeError(f"Expected a float but got a {type(value)}.")
        os.environ[self.__env_name] = str(value)

    def get(self) -> float:
        return float(os.environ[self.__env_name])

    def reset(self) -> None:
        os.environ[self.__env_name] = str(self.__default_value)
