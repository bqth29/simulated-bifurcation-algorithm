import pytest
from src.simulated_bifurcation.optimizer.optimization_variables import OptimizationVariable, get_env


def test_optimization_variables_initialized():
    assert .1 == OptimizationVariable.TIME_STEP.get()
    assert .01 == OptimizationVariable.PRESSURE_SLOPE.get()
    assert .06 == OptimizationVariable.HEAT_COEFFICIENT.get()

def test_set_and_reset_variable():
    OptimizationVariable.TIME_STEP.set(.05)
    assert .05 == OptimizationVariable.TIME_STEP.get()
    OptimizationVariable.TIME_STEP.reset()
    assert .1 == OptimizationVariable.TIME_STEP.get()

def test_set_variable_with_wrong_type_throws_error():
    with pytest.raises(TypeError):
       OptimizationVariable.TIME_STEP.set('Hello world!')

def test_get_env():
    assert {'TIME_STEP': .1, 'PRESSURE_SLOPE': .01, 'HEAT_COEFFICIENT': .06} == get_env()
