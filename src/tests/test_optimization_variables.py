import pytest

from src.simulated_bifurcation.optimizer.optimization_variables import (
    OptimizationVariable,
    get_env,
)


def test_optimization_variables_initialized():
    assert 0.1 == OptimizationVariable.TIME_STEP.get()
    assert 0.01 == OptimizationVariable.PRESSURE_SLOPE.get()
    assert 0.06 == OptimizationVariable.HEAT_COEFFICIENT.get()


def test_set_and_reset_variable():
    OptimizationVariable.TIME_STEP.set(0.05)
    assert 0.05 == OptimizationVariable.TIME_STEP.get()
    OptimizationVariable.TIME_STEP.reset()
    assert 0.1 == OptimizationVariable.TIME_STEP.get()


def test_set_variable_with_wrong_type_throws_error():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        OptimizationVariable.TIME_STEP.set("Hello world!")


def test_get_env():
    assert {
        "TIME_STEP": 0.1,
        "PRESSURE_SLOPE": 0.01,
        "HEAT_COEFFICIENT": 0.06,
    } == get_env()
