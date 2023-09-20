import torch

from src.simulated_bifurcation import set_env
from src.simulated_bifurcation.models import Knapsack


def test_markowitz():
    torch.manual_seed(42)
    weights = [12, 1, 1, 4, 2]
    prices = [4, 2, 1, 10, 2]
    model = Knapsack(weights, prices, max_weight=15)

    assert model.summary == {
        "items": [],
        "total_cost": 0,
        "total_weight": 0,
        "status": "not optimized",
    }

    model.minimize(verbose=False)
    assert model.summary["items"] == [1, 2, 3, 4]
    assert model.summary["total_cost"] == 15
    assert model.summary["total_weight"] == 8
    assert model.summary["status"] == "success"
