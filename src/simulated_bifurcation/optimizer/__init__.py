"""
Implementation of the Simulated Bifurcation algorithm. This modules serves as
a back-end of the `core` module which helps define optimization problems to be
solved with the Simulated Bifurcation algorithm.

See Also
--------
core:
    Module of utility models to help define and solve optimization
    problems with the Simulated Bifurcation algorithm.
models:
    Package containing the implementation of several common combinatorial
    optimization problems.

"""

from .environment import get_env, reset_env, set_env
from .simulated_bifurcation_engine import SimulatedBifurcationEngine
from .simulated_bifurcation_optimizer import (
    ConvergenceWarning,
    SimulatedBifurcationOptimizer,
)
from .stop_window import StopWindow
from .symplectic_integrator import SymplecticIntegrator
