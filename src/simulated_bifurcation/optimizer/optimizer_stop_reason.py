from enum import Enum


class OptimizerStopReason(Enum):
    STEPS = "Maximum number of steps reached."
    TIMEOUT = "Maximum simulation time reached."
    WINDOW = "All agents converged."

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message
