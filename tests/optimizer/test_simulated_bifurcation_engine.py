from src.simulated_bifurcation.optimizer.simulated_bifurcation_engine import (
    SimulatedBifurcationEngine,
)


def test_simulated_bifurcation_engine():
    assert SimulatedBifurcationEngine.bSB == SimulatedBifurcationEngine.get_engine(
        ballistic=True, heated=False
    )
    assert SimulatedBifurcationEngine.dSB == SimulatedBifurcationEngine.get_engine(
        ballistic=False, heated=False
    )
    assert SimulatedBifurcationEngine.HbSB == SimulatedBifurcationEngine.get_engine(
        ballistic=True, heated=True
    )
    assert SimulatedBifurcationEngine.HdSB == SimulatedBifurcationEngine.get_engine(
        ballistic=False, heated=True
    )
