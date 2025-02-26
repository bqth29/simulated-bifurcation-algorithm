import numpy as np
import torch

from src.simulated_bifurcation.models import SequentialMarkowitz

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
    model = SequentialMarkowitz(
        covariances,
        expected_returns,
        np.array(
            [
                [[0.1, 0.0], [0.0, 0.2]],
                [[0.11, 0.0], [0.0, 0.2]],
                [[0.05, 0.0], [0.0, 0.4]],
                [[0.01, 0.0], [0.0, 0.5]],
            ]
        ),
        initial_stocks,
        risk_coefficient=1,
        number_of_bits=1,
        dtype=torch.float32,
    )

    assert torch.equal(covariances, model.covariances)
    assert torch.equal(expected_returns, model.expected_returns)
    assert torch.equal(rebalancing_costs, model.rebalancing_costs)
    assert torch.equal(initial_stocks, model.initial_stocks)
    assert model.portfolio is None
    assert model.gains is None

    assert torch.all(
        torch.isclose(
            torch.tensor(
                [
                    [-0.71, -0.10, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.10, -0.90, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0],
                    [0.11, 0.0, -0.66, -0.15, 0.05, 0.0, 0.0, 0.0],
                    [0.0, 0.20, -0.15, -1.10, 0.0, 0.40, 0.0, 0.0],
                    [0.0, 0.0, 0.05, 0.0, -0.56, -0.175, 0.01, 0.0],
                    [0.0, 0.0, 0.0, 0.40, -0.175, -1.40, 0.0, 0.50],
                    [0.0, 0.0, 0.0, 0.0, 0.01, 0.0, -0.51, 0.06],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.06, -1.0],
                ]
            ),
            model.quadratic,
        )
    )
    assert torch.all(
        torch.isclose(
            torch.tensor(
                [0.5000, 1.0000, 0.5500, 0.5500, 0.7000, 0.5000, 1.4000, 0.2000]
            ),
            model.linear,
        )
    )
    assert torch.equal(torch.tensor(-0.2), model.bias)

    model.maximize(agents=128, early_stopping=False, verbose=False)
    assert (4, 2) == model.portfolio.shape
    assert torch.equal(
        torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), model.portfolio
    )
    assert 0.95 == round(model.gains, 4)
