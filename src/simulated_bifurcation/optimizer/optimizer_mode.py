from enum import Enum

import torch


class OptimizerMode(Enum):
    BALLISTIC = torch.nn.Identity()
    DISCRETE = torch.sign

    def __init__(self, activation_function) -> None:
        super().__init__()
        self.activation_function = activation_function
