from typing import Optional, Tuple

import torch


class Preprocessing:
    def __init__(self, J: torch.Tensor, h: torch.Tensor) -> None:
        self.J = J.clone()
        self.h = h.clone()
        self.optimized_spins = torch.zeros(self.J.shape[0])
        self.shifted_indices = list(range(self.J.shape[0]))

    def _remove_row(self, index: int):
        self.J = torch.cat((self.J[:index], self.J[index + 1 :]), dim=0)
        self.h = torch.cat((self.h[:index], self.h[index + 1 :]))

    def _remove_column(self, index: int):
        self.J = torch.cat((self.J[:, :index], self.J[:, index + 1 :]), dim=1)

    def _remove_all_coefficients(self, index: int):
        self._remove_row(index)
        self._remove_column(index)

    def _project_coefficients_in_linear_part(self, row_index: int, sign: int):
        self.h -= sign * self.J[row_index]

    def _project_coefficients_and_delete_row_and_column(
        self, spin_index: int, spin_sign: int
    ):
        self._project_coefficients_in_linear_part(spin_index, spin_sign)
        self._remove_all_coefficients(spin_index)

    def _get_optimizable_spins(self) -> torch.Tensor:
        return torch.abs(self.h) >= torch.sum(torch.abs(self.J), dim=0)

    def _get_first_optimizable_spin(self) -> Optional[int]:
        optimizable_spins = self._get_optimizable_spins()
        if torch.any(optimizable_spins).item():
            return torch.nonzero(optimizable_spins)[0].item()
        return None

    def _delete_index(self, index: int):
        self.shifted_indices = (
            self.shifted_indices[:index] + self.shifted_indices[index + 1 :]
        )

    def _get_original_index(self, new_index: int) -> int:
        return self.shifted_indices[new_index]

    def _set_spin_value(self, spin_index: int, spin_sign: int):
        self.optimized_spins[spin_index] = spin_sign

    def _set_first_optimal_spin(self) -> bool:
        optimal_spin_index = self._get_first_optimizable_spin()
        if optimal_spin_index is None:
            return False
        sign = -1 if self.h[optimal_spin_index] > 0 else 1
        original_index = self._get_original_index(optimal_spin_index)
        self._set_spin_value(original_index, sign)
        self._project_coefficients_and_delete_row_and_column(optimal_spin_index, sign)
        self._delete_index(optimal_spin_index)
        return True

    def presolve(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _continue = True
        while _continue:
            optimized_spin = self._set_first_optimal_spin()
            _continue = _continue and optimized_spin and self.J.shape[1] > 0
        return self.optimized_spins.clone(), self.J.clone(), self.h.clone()
