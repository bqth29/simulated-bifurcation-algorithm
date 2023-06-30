import pytest
from torch import float32
from src.simulated_bifurcation.polynomial import IsingInterface


def test_cannot_init_ising_interface():
    with pytest.raises(TypeError):
        IsingInterface(float32, 'cpu')
