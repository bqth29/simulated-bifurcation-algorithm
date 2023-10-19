import pytest
from torch import device, float32, float64

from src.simulated_bifurcation import get_env, reset_env, set_env


def test_get_env():
    assert {
        "TIME_STEP": 0.1,
        "PRESSURE_SLOPE": 0.01,
        "HEAT_COEFFICIENT": 0.06,
        "DEFAULT_DTYPE": float32,
        "DEFAULT_DEVICE": device("cpu"),
    } == get_env()


def test_set_env():
    set_env(
        time_step=0.05,
        pressure_slope=0.1,
        heat_coefficient=0.08,
        default_dtype=float64,
        default_device="cuda:0",
    )
    assert {
        "TIME_STEP": 0.05,
        "PRESSURE_SLOPE": 0.1,
        "HEAT_COEFFICIENT": 0.08,
        "DEFAULT_DTYPE": float64,
        "DEFAULT_DEVICE": device("cuda:0"),
    } == get_env()
    reset_env()


def test_set_env_with_wrong_parameters():
    with pytest.raises(TypeError, match="All optimization variables must be floats."):
        # noinspection PyTypeChecker
        set_env(time_step="Hello world!")
    with pytest.raises(TypeError, match="Default dtype must be a valid torch dtype."):
        # noinspection PyTypeChecker
        set_env(default_dtype="Hello world!")
    with pytest.raises(TypeError, match="Default device must be a string."):
        # noinspection PyTypeChecker
        set_env(default_device=0.1)
    with pytest.raises(ValueError, match="Invalid device type."):
        set_env(default_device="Hello world!")
