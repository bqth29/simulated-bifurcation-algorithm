import torch

from src.simulated_bifurcation.models import NumberPartitioning


def test_number_partitioning_with_even_sum():
    torch.manual_seed(42)
    numbers = [2, 7, 3, 8, 1, 5, 4, 6]
    model = NumberPartitioning(numbers, dtype=torch.float32)
    model.optimize(agents=10, verbose=False)
    result = model.partition
    assert result["left"]["sum"] == result["right"]["sum"]


def test_number_partitioning_with_odd_sum():
    torch.manual_seed(42)
    numbers = [2, 7, 3, 9, 1, 5, 4, 6]
    model = NumberPartitioning(numbers, dtype=torch.float32)
    model.optimize(agents=10, verbose=False)
    result = model.partition
    assert abs(result["left"]["sum"] - result["right"]["sum"]) == 1


def test_not_optimized_model():
    model = NumberPartitioning([1, 2, 3])
    assert model.partition == {
        "left": {"values": [], "sum": None},
        "right": {"values": [], "sum": None},
    }
