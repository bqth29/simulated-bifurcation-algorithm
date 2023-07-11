import pytest
from src.simulated_bifurcation.polynomial import IsingInterface


def test_ising_interface():
    with pytest.raises(NotImplementedError):
        IsingInterface.to_ising(None)
    with pytest.raises(NotImplementedError):
        IsingInterface.from_ising(None, None)
