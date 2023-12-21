from typing import Dict, Optional


class Environment:
    def __init__(self) -> None:
        self.reset()

    def set_time_step(self, time_step: float):
        self.__check_float_else_throw(time_step)
        self.time_step = time_step

    def set_pressure_slope(self, pressure_slope: float):
        self.__check_float_else_throw(pressure_slope)
        self.pressure_slope = pressure_slope

    def set_heat_coefficient(self, heat_coefficient: float):
        self.__check_float_else_throw(heat_coefficient)
        self.heat_coefficient = heat_coefficient

    def reset(self):
        """
        Sets the default values of the SB optimization variables.
        """
        self.time_step = 0.1
        self.pressure_slope = 0.01
        self.heat_coefficient = 0.06

    def as_dict(self) -> Dict[str, float]:
        """
        Returns the values of the SB optimization variables as a dictionnary.
        """
        return {
            "time_step": self.time_step,
            "pressure_slope": self.pressure_slope,
            "heat_coefficient": self.heat_coefficient,
        }

    def __check_float_else_throw(self, value: float):
        if not isinstance(value, float):
            raise TypeError(f"Expected a float but got a {type(value)}.")


ENVIRONMENT = Environment()


def get_env() -> Dict[str, float]:
    """
    Returns the values of the quantum physical constants behind the
    Simulated Bifurcation algorithm.
    """
    return ENVIRONMENT.as_dict()


def set_env(
    *,
    time_step: Optional[float] = None,
    pressure_slope: Optional[float] = None,
    heat_coefficient: Optional[float] = None,
):
    """
    Override the values of the pre-set and fine-tuned quantum physical
    constants behind the Simulated Bifurcation algorithm.

    Parameters
    ----------
    time_step: float, optional
        Temporal discretization step.
    pressure_slope: float, optional
        Adiabatic system evolution rate.
    heat_coefficient: float, optional
        Influence of heating for HbSB or HdSB.

    Notes
    -----
    All parameters are keyword-only and optional with a default
    value to `None`. `None` means that the variable is not
    changed in the environment.

    To set a default dtype and a default device for tensors
    please use `torch.set_default_dtype` and `torch.set_default_device`.
    """
    if time_step is not None:
        ENVIRONMENT.set_time_step(time_step)
    if pressure_slope is not None:
        ENVIRONMENT.set_pressure_slope(pressure_slope)
    if heat_coefficient is not None:
        ENVIRONMENT.set_heat_coefficient(heat_coefficient)


def reset_env() -> None:
    """
    Reset the Simulated Bifurcation algorithm quantum physical
    constants to their original fine-tuned value:

    - time step: 0.1
    - pressure slope: 0.01
    - heat coefficient: 0.06
    """
    ENVIRONMENT.reset()
