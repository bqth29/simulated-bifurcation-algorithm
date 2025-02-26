from enum import Enum
from typing import Callable, Literal

import torch


class SimulatedBifurcationEngine(Enum):
    """
    Variants of the Simulated Bifurcation algorithm.
    """

    bSB = torch.nn.Identity()
    dSB = torch.sign

    def __init__(
        self, activation_function: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.__activation_function = activation_function

    @property
    def activation_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.__activation_function

    @staticmethod
    def get_engine(engine_name: Literal["ballistic", "discrete"]):
        if engine_name == "ballistic":
            return SimulatedBifurcationEngine.bSB
        elif engine_name == "discrete":
            return SimulatedBifurcationEngine.dSB
        else:
            raise ValueError(f"Unknwown Simulated Bifurcation engine: {engine_name}.")
