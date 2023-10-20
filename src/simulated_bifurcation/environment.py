from typing import Dict, Optional

from .optimizer.optimization_variables import OptimizationVariable


def get_env() -> Dict[str, float]:
    """
    Returns the values of the quantum physical constants behind the
    Simulated Bifurcation algorithm.
    """
    return {variable.name: variable.get() for variable in OptimizationVariable}


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

    See Also
    --------
    To set a default dtype and a default device for tensors
    please use `torch.set_default_dtype` and
    `torch.set_default_device`.
    """
    if (
        (time_step is None or isinstance(time_step, float))
        and (pressure_slope is None or isinstance(pressure_slope, float))
        and (heat_coefficient is None or isinstance(heat_coefficient, float))
    ):
        OptimizationVariable.TIME_STEP.set(time_step)
        OptimizationVariable.PRESSURE_SLOPE.set(pressure_slope)
        OptimizationVariable.HEAT_COEFFICIENT.set(heat_coefficient)
    else:
        raise TypeError("All optimization variables must be floats.")


def reset_env() -> None:
    """
    Reset the Simulated Bifurcation algorithm quantum physical
    constants to their original fine-tuned value:

    - time step: 0.1
    - pressure slope: 0.01
    - heat coefficient: 0.06
    """
    for variable in OptimizationVariable:
        variable.reset()
