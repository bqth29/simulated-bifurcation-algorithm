import torch
from enum import Enum


class OptimizerMode(Enum):
    BALLISTIC = torch.nn.Identity()
    DISCRETE = torch.sign

    @property
    def activation_function(self) -> str: return self.value
