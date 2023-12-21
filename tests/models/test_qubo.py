import torch

from src.simulated_bifurcation.models import QUBO


def test_qubo():
    torch.manual_seed(42)
    Q = torch.tensor([[1, -2, 3], [0, -4, 1], [0, 0, 2]])
    model = QUBO(Q, dtype=torch.float32)
    binary_vector, value = model.minimize(agents=10, verbose=False, best_only=True)
    assert torch.equal(torch.tensor([1.0, 1.0, 0.0]), binary_vector)
    assert -5.0 == value
    binary_vector, value = model.maximize(agents=10, verbose=False, best_only=True)
    assert torch.equal(torch.tensor([1.0, 0.0, 1.0]), binary_vector)
    assert 6.0 == value


def test_lp_problem_formulated_as_qubo():
    torch.manual_seed(42)
    P = 15.5
    Q = torch.tensor(
        [
            [2, -P, -P, 0, 0, 0],
            [0, 2, -P, -P, 0, 0],
            [0, 0, 2, -2 * P, 0, 0],
            [0, 0, 0, 2, -P, 0],
            [0, 0, 0, 0, 4.5, -P],
            [0, 0, 0, 0, 0, 3],
        ]
    )
    binary_vector, objective_value = QUBO(Q, dtype=torch.float32).maximize(
        agents=10, verbose=False
    )
    assert torch.equal(
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32), binary_vector
    )
    assert 7.0 == objective_value
