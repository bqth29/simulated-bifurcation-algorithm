from enum import Enum
from typing import Callable

import torch


class SimulatedBifurcationEngine(Enum):
    """
    Enum class that gathers the 4 variants of the Simulated Bifurcation
    algorithm:

    1. Ballistic SB (bSB)
    2. Discrete SB (dSB)
    3. Heated ballistic SB (HbSB)
    4. Heated discrete SB (HdSB)
    """

    bSB = torch.nn.Identity(), False
    dSB = torch.sign, False
    HbSB = torch.nn.Identity(), True
    HdSB = torch.sign, True

    def __init__(
        self, activation_function: Callable[[torch.Tensor], torch.Tensor], heated: bool
    ) -> None:
        self.activation_function = activation_function
        self.heated = heated

    @staticmethod
    def get_engine(ballistic: bool, heated: bool):
        if ballistic:
            if heated:
                return SimulatedBifurcationEngine.HbSB
            return SimulatedBifurcationEngine.bSB
        if heated:
            return SimulatedBifurcationEngine.HdSB
        return SimulatedBifurcationEngine.dSB
