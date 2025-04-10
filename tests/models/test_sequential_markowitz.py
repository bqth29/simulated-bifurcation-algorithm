import torch

from src.simulated_bifurcation.models import sequential_markowitz

#  2 assets on 4 timestamps case test
initial_stocks = torch.tensor([0.0, 1.0])
covariances = torch.tensor(
    [
        [[1.0, 0.2], [0.2, 1]],
        [[1.0, 0.3], [0.3, 1]],
        [[1.0, 0.35], [0.35, 1]],
        [[1.0, -0.12], [-0.12, 1]],
    ]
)
expected_returns = torch.tensor(
    [
        [0.5, 0.6],
        [0.55, 0.55],
        [0.7, 0.5],
        [1.4, 0.2],
    ]
)
rebalancing_costs = torch.tensor(
    [
        [[0.1, 0.0], [0.0, 0.2]],
        [[0.11, 0.0], [0.0, 0.2]],
        [[0.05, 0.0], [0.0, 0.4]],
        [[0.01, 0.0], [0.0, 0.5]],
    ]
)


def test_sequential_markowitz():
    torch.manual_seed(42)
    model = sequential_markowitz(
        covariances,
        expected_returns,
        rebalancing_costs,
        initial_stocks,
        risk_coefficient=1,
        number_of_bits=1,
    )

    result = model.solve(agents=128, early_stopping=False, verbose=False)
    assert (4, 2) == result[0].shape
    assert torch.equal(
        torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), result[0]
    )
    assert 0.95 == round(result[1], 2)
