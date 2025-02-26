import pytest

from src.simulated_bifurcation.optimizer.simulated_bifurcation_engine import (
    SimulatedBifurcationEngine,
)


def test_simulated_bifurcation_engine():
    assert SimulatedBifurcationEngine.bSB == SimulatedBifurcationEngine.get_engine(
        "ballistic"
    )
    assert SimulatedBifurcationEngine.dSB == SimulatedBifurcationEngine.get_engine(
        "discrete"
    )
    with pytest.raises(
        ValueError, match="Unknwown Simulated Bifurcation engine: unknown-engine."
    ):
        SimulatedBifurcationEngine.get_engine("unknown-engine")
