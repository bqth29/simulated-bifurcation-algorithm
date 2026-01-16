import torch

from src.simulated_bifurcation.models import number_partitioning


def test_number_partitioning_with_even_sum():
    torch.manual_seed(42)
    numbers = [2, 7, 3, 8, 1, 5, 4, 6]
    model = number_partitioning(numbers)
    result = model.solve(agents=10, verbose=False, dtype=torch.float32)
    assert result["left"]["sum"] == result["right"]["sum"]


def test_number_partitioning_with_odd_sum():
    torch.manual_seed(42)
    numbers = [2, 7, 3, 9, 1, 5, 4, 6]
    model = number_partitioning(numbers)
    result = model.solve(agents=10, verbose=False, dtype=torch.float32)
    assert abs(result["left"]["sum"] - result["right"]["sum"]) == 1
