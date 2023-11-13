import torch

from src.simulated_bifurcation.models import Markowitz


def test_markowitz():
    torch.manual_seed(42)
    covariance = torch.tensor([[1.0, 1.2, 0.7], [1.2, 1.0, -1.9], [0.7, -1.9, 1.0]])
    expected_return = torch.tensor([0.2, 0.05, 0.17])
    model = Markowitz(
        covariance,
        expected_return,
        risk_coefficient=1,
        number_of_bits=3,
        dtype=torch.float32,
    )

    assert torch.all(torch.isclose(covariance, model.covariance))
    assert torch.equal(expected_return, model.expected_return)
    assert model.portfolio is None
    assert model.gains is None

    model.maximize(agents=10, verbose=False)
    assert torch.equal(torch.tensor([0.0, 7.0, 7.0]), model.portfolio)
    assert 45.64 == round(model.gains, 2)
