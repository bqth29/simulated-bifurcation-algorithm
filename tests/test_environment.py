import os

import pytest

from src.simulated_bifurcation import get_env, reset_env, set_env


def test_get_env():
    assert {
        "TIME_STEP": 0.1,
        "PRESSURE_SLOPE": 0.01,
        "HEAT_COEFFICIENT": 0.06,
    } == get_env()
    assert float(os.environ.get("PYTHON_SB_TIME_STEP")) == 0.1
    assert float(os.environ.get("PYTHON_SB_PRESSURE_SLOPE")) == 0.01
    assert float(os.environ.get("PYTHON_SB_HEAT_COEFFICIENT")) == 0.06


def test_set_env():
    set_env(
        time_step=0.05,
        pressure_slope=0.1,
        heat_coefficient=0.08,
    )
    assert {
        "TIME_STEP": 0.05,
        "PRESSURE_SLOPE": 0.1,
        "HEAT_COEFFICIENT": 0.08,
    } == get_env()
    reset_env()


def test_set_env_none():
    set_env(pressure_slope=0.2)
    set_env(heat_coefficient=0.04)
    assert {
        "TIME_STEP": 0.1,
        "PRESSURE_SLOPE": 0.2,
        "HEAT_COEFFICIENT": 0.04,
    } == get_env()
    reset_env()


def test_set_env_with_wrong_parameters():
    with pytest.raises(TypeError, match="All optimization variables must be floats."):
        # noinspection PyTypeChecker
        set_env(time_step="Hello world!")
    with pytest.raises(TypeError, match="All optimization variables must be floats."):
        # noinspection PyTypeChecker
        set_env(pressure_slope="Hello world!")
    with pytest.raises(TypeError, match="All optimization variables must be floats."):
        # noinspection PyTypeChecker
        set_env(heat_coefficient="Hello world!")
