import os

import pytest

from src.simulated_bifurcation import get_env, reset_env, set_env


def test_get_env():
    assert {
        "time_step": 0.1,
        "pressure_slope": 0.01,
        "heat_coefficient": 0.06,
    } == get_env()


def test_set_env():
    set_env(
        time_step=0.05,
        pressure_slope=0.1,
        heat_coefficient=0.08,
    )
    assert {
        "time_step": 0.05,
        "pressure_slope": 0.1,
        "heat_coefficient": 0.08,
    } == get_env()
    reset_env()


def test_set_env_none():
    set_env(pressure_slope=0.2)
    set_env(heat_coefficient=0.04)
    assert {
        "time_step": 0.1,
        "pressure_slope": 0.2,
        "heat_coefficient": 0.04,
    } == get_env()
    reset_env()


def test_set_env_with_wrong_parameters():
    with pytest.raises(TypeError, match="Expected a float but got a <class 'str'>."):
        # noinspection PyTypeChecker
        set_env(time_step="Hello world!")
    with pytest.raises(TypeError, match="Expected a float but got a <class 'str'>."):
        # noinspection PyTypeChecker
        set_env(pressure_slope="Hello world!")
    with pytest.raises(TypeError, match="Expected a float but got a <class 'str'>."):
        # noinspection PyTypeChecker
        set_env(heat_coefficient="Hello world!")
