import torch

from src.simulated_bifurcation.models import SequentialMarkowitz

#  2 assets on 4 timestamps case test
initial_stocks = torch.tensor([1.0, 0.0])
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
        [[0.1, 0.4], [0.4, 0.2]],
        [[0.11, 0.38], [0.38, 0.2]],
        [[0.05, 0.32], [0.32, 0.4]],
        [[0.01, 0.4], [0.4, 0.5]],
    ]
)


def test_sequential_markowitz():
    torch.manual_seed(42)
    model = SequentialMarkowitz(
        covariances,
        expected_returns,
        rebalancing_costs,
        initial_stocks,
        risk_coefficient=1,
        number_of_bits=3,
    )

    assert torch.equal(covariances, model.covariances)
    assert torch.equal(expected_returns, model.expected_returns)
    assert torch.equal(rebalancing_costs, model.rebalancing_costs)
    assert torch.equal(initial_stocks, model.initial_stocks)
    assert model.portfolio is None
    assert model.gains == 0.0

    assert torch.all(
        torch.isclose(
            torch.tensor(
                [
                    [-0.71, -0.88, 0.11, 0.38, 0.0, 0.0, 0.0, 0.0],
                    [-0.88, -0.90, 0.38, 0.20, 0.0, 0.0, 0.0, 0.0],
                    [0.11, 0.38, -0.66, -0.85, 0.05, 0.32, 0.0, 0.0],
                    [0.38, 0.20, -0.85, -1.10, 0.32, 0.40, 0.0, 0.0],
                    [0.0, 0.0, 0.05, 0.32, -0.56, -0.895, 0.01, 0.40],
                    [0.0, 0.0, 0.32, 0.40, -0.895, -1.40, 0.40, 0.50],
                    [0.0, 0.0, 0.0, 0.0, 0.01, 0.40, -0.51, -0.34],
                    [0.0, 0.0, 0.0, 0.0, 0.40, 0.50, -0.34, -1.0],
                ]
            ),
            model.matrix,
        )
    )
    assert torch.all(
        torch.isclose(
            torch.tensor(
                [0.7000, 1.4000, 0.5500, 0.5500, 0.7000, 0.5000, 1.4000, 0.2000]
            ),
            model.vector,
        )
    )
    assert torch.equal(torch.tensor(-0.1), model.constant)

    model.maximize(agents=20, use_window=False, verbose=False)
    assert (4, 2) == model.portfolio.shape
    assert torch.equal(
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), model.portfolio
    )
    assert 0.79 == round(model.gains, 4)