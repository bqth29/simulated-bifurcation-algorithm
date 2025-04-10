import torch

from src.simulated_bifurcation.models import markowitz


def test_markowitz():
    torch.manual_seed(42)
    covariance = torch.tensor([[1.0, 1.2, 0.7], [1.2, 1.0, -1.9], [0.7, -1.9, 1.0]])
    expected_return = torch.tensor([0.2, 0.05, 0.17])
    model = markowitz(
        covariance,
        expected_return,
        risk_coefficient=1,
        number_of_bits=3,
    )

    result = model.solve(agents=10, verbose=False)
    assert torch.equal(torch.tensor([0.0, 7.0, 7.0]), result[0])
    assert 45.64 == round(result[1], 2)
