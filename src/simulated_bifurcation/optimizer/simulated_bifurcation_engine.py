from enum import Enum

import torch


class SimulatedBifurcationEngine(Enum):
    bSB = torch.nn.Identity(), False
    dSB = torch.sign, False
    HbSB = torch.nn.Identity(), True
    HdSB = torch.sign, True

    def __init__(self, activation_function, heated: bool) -> None:
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
