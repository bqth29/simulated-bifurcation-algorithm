import pytest
import torch

from src.simulated_bifurcation.optimizer.preprocessing import Preprocessing

J = torch.tensor(
    [
        [1, 2, 3],
        [2, 1, 4],
        [3, 4, 1],
    ],
    dtype=torch.float32,
)
h = torch.tensor([1, 0, -2], dtype=torch.float32)

optimizable_J = torch.tensor(
    [
        [0.0, 0.5, -1.0],
        [0.5, 0.0, 2.0],
        [-1.0, 2.0, 0.0],
    ],
    dtype=torch.float32,
)
optimizable_h = torch.tensor([2.0, 1.0, -4.0], dtype=torch.float32)


def test_remove_row():
    preprocesser = Preprocessing(J, h)
    preprocesser._remove_row(1)
    assert torch.equal(
        preprocesser.J, torch.tensor([[1, 2, 3], [3, 4, 1]], dtype=torch.float32)
    )
    assert torch.equal(preprocesser.h, torch.tensor([1, -2], dtype=torch.float32))
    preprocesser._remove_row(0)
    assert torch.equal(preprocesser.J, torch.tensor([[3, 4, 1]], dtype=torch.float32))
    assert torch.equal(preprocesser.h, torch.tensor([-2], dtype=torch.float32))


def test_remove_column():
    preprocesser = Preprocessing(J, h)
    preprocesser._remove_column(1)
    assert torch.equal(
        preprocesser.J, torch.tensor([[1, 3], [2, 4], [3, 1]], dtype=torch.float32)
    )
    assert torch.equal(preprocesser.h, h)
    preprocesser._remove_column(0)
    assert torch.equal(
        preprocesser.J, torch.tensor([[3], [4], [1]], dtype=torch.float32)
    )
    assert torch.equal(preprocesser.h, h)


def test_remove_all_coefficients():
    preprocesser = Preprocessing(J, h)
    preprocesser._remove_all_coefficients(1)
    assert torch.equal(
        preprocesser.J, torch.tensor([[1, 3], [3, 1]], dtype=torch.float32)
    )
    assert torch.equal(preprocesser.h, torch.tensor([1, -2], dtype=torch.float32))
    preprocesser._remove_all_coefficients(0)
    assert torch.equal(preprocesser.J, torch.tensor([[1]], dtype=torch.float32))
    assert torch.equal(preprocesser.h, torch.tensor([-2], dtype=torch.float32))


def test_project_coefficients_in_linear_part():
    preprocesser = Preprocessing(J, h)
    preprocesser._project_coefficients_in_linear_part(1, 1)
    assert torch.equal(preprocesser.J, J)
    assert torch.equal(preprocesser.h, torch.tensor([-1, -1, -6], dtype=torch.float32))
    preprocesser._project_coefficients_in_linear_part(0, -1)
    assert torch.equal(preprocesser.J, J)
    assert torch.equal(preprocesser.h, torch.tensor([0, 1, -3], dtype=torch.float32))


def test_project_coefficients_and_delete_row_and_column():
    preprocesser = Preprocessing(J, h)
    preprocesser._project_coefficients_and_delete_row_and_column(2, -1)
    assert torch.equal(
        preprocesser.J, torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)
    )
    assert torch.equal(preprocesser.h, torch.tensor([4, 4], dtype=torch.float32))


def test_get_optimizable_spins():
    preprocesser = Preprocessing(optimizable_J, optimizable_h)
    assert torch.equal(
        preprocesser._get_optimizable_spins(), torch.Tensor([True, False, True])
    )


def test_get_first_optimizable_spin():
    optimizable_preprocesser = Preprocessing(optimizable_J, optimizable_h)
    assert 0 == optimizable_preprocesser._get_first_optimizable_spin()
    non_optimizable_preprocesser = Preprocessing(J, h)
    assert non_optimizable_preprocesser._get_first_optimizable_spin() is None


def test_delete_index():
    preprocesser = Preprocessing(J, h)
    assert [0, 1, 2] == preprocesser.shifted_indices
    preprocesser._delete_index(1)
    assert [0, 2] == preprocesser.shifted_indices
    preprocesser._delete_index(0)
    assert [2] == preprocesser.shifted_indices


def test_get_original_index():
    preprocesser = Preprocessing(J, h)
    preprocesser._delete_index(1)
    assert 0 == preprocesser._get_original_index(0)
    assert 2 == preprocesser._get_original_index(1)


def test_set_spin_value():
    preprocesser = Preprocessing(J, h)
    assert torch.equal(preprocesser.optimized_spins, torch.tensor([0, 0, 0]))
    preprocesser._set_spin_value(2, -1)
    assert torch.equal(preprocesser.optimized_spins, torch.tensor([0, 0, -1]))
    preprocesser._set_spin_value(0, 1)
    assert torch.equal(preprocesser.optimized_spins, torch.tensor([1, 0, -1]))


def test_set_first_optimal_spin():
    non_optimizable_preprocesser = Preprocessing(J, h)
    assert non_optimizable_preprocesser._set_first_optimal_spin() is False
    optimizable_preprocesser = Preprocessing(optimizable_J, optimizable_h)
    assert optimizable_preprocesser._set_first_optimal_spin() is True
    assert torch.equal(
        optimizable_preprocesser.J, torch.tensor([[0, 2], [2, 0]], dtype=torch.float32)
    )
    assert torch.equal(
        optimizable_preprocesser.h, torch.tensor([1.5, -5], dtype=torch.float32)
    )
    assert torch.equal(
        optimizable_preprocesser.optimized_spins, torch.tensor([-1, 0, 0])
    )
    assert optimizable_preprocesser.shifted_indices == [1, 2]


def test_presolve():
    non_optimizable_preprocesser = Preprocessing(J, h)
    (
        non_optimized_spins,
        non_optimized_J,
        non_optimized_h,
    ) = non_optimizable_preprocesser.presolve()
    assert torch.equal(non_optimized_spins, torch.tensor([0, 0, 0]))
    assert torch.equal(non_optimized_J, J)
    assert torch.equal(non_optimized_h, h)
    optimizable_preprocesser = Preprocessing(optimizable_J, optimizable_h)
    optimized_spins, optimized_J, optimized_h = optimizable_preprocesser.presolve()
    assert torch.equal(optimized_spins, torch.tensor([-1, 1, 1]))
    assert torch.equal(optimized_J, torch.tensor([]).reshape(0, 0))
    assert torch.equal(optimized_h, torch.tensor([]))
