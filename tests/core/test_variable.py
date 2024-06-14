import pytest

from simulated_bifurcation.core.optimization_domain import OptimizationDomain
from simulated_bifurcation.core.variable import Variable


def test_init_spin_variable():
    spin_variable = Variable(OptimizationDomain.SPIN, 1)
    assert spin_variable.is_spin
    assert spin_variable.encoding_bits == 1


def test_init_spin_variable_from_string():
    spin_variable = Variable.from_str("spin")
    assert spin_variable.is_spin
    assert spin_variable.encoding_bits == 1


def test_init_spin_variable_with_sevral_encoding_bits():
    with pytest.raises(
        ValueError, match="A spin or binary variable can only be encoded with one bit."
    ):
        Variable(OptimizationDomain.SPIN, 2)


def test_init_binary_variable():
    binary_variable = Variable(OptimizationDomain.BINARY, 1)
    assert not binary_variable.is_spin
    assert binary_variable.encoding_bits == 1


def test_init_binary_variable_from_string():
    binary_variable = Variable.from_str("binary")
    assert not binary_variable.is_spin
    assert binary_variable.encoding_bits == 1


def test_init_binary_variable_with_sevral_encoding_bits():
    with pytest.raises(
        ValueError, match="A spin or binary variable can only be encoded with one bit."
    ):
        Variable(OptimizationDomain.BINARY, 2)


@pytest.mark.parametrize("encoding_bits", [1, 3, 17, 42, 100])
def test_init_integer_variable(encoding_bits: int):
    int_variable = Variable(OptimizationDomain.INTEGER, encoding_bits)
    assert not int_variable.is_spin
    assert int_variable.encoding_bits == encoding_bits


@pytest.mark.parametrize("encoding_bits", [1, 3, 17, 42, 100])
def test_init_integer_variable_from_string(encoding_bits: int):
    int_variable = Variable.from_str(f"int{encoding_bits}")
    assert not int_variable.is_spin
    assert int_variable.encoding_bits == encoding_bits


def test_init_integer_variable_from_wrong_string_pattern():
    expected_error_message = (
        r"Domain type must be one of \"spin\" or \"binary\", or be a string starting "
        r"with \"int\" and followed by a positive integer that represents "
        r"the number of bits required to encode the values of the domain\. "
        r"More formally, it should match the following regular expression: "
        r"\"\^int\[1-9\]\[0-9\]\*\$\" \(ex: \"int7\", \"int42\", \.\.\.\)\."
    )
    with pytest.raises(ValueError, match=expected_error_message):
        Variable.from_str("int0")
    with pytest.raises(ValueError, match=expected_error_message):
        Variable.from_str("int1.5")
    with pytest.raises(ValueError, match=expected_error_message):
        Variable.from_str("Hello world!")
