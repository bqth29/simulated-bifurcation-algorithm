from typing import Optional, Tuple

import torch


class Preprocessing:
    """
    Utilility class to presolve Ising models. It identifies spins
    the optimal sign of which can be determined before running the
    Simulated Bifurcation algorithm in order to only use a reduced
    version of the original Ising model, saving computation resources
    and time.
    """

    def __init__(self, J: torch.Tensor, h: torch.Tensor) -> None:
        self.J = J.clone()
        self.h = h.clone()
        self.optimized_spins = torch.zeros(self.J.shape[0])
        self.shifted_indices = list(range(self.J.shape[0]))

    def _remove_row(self, index: int):
        """
        Remove a given row from both the matrix J and the
        vector h.

        Parameters
        ----------
        index : int
            Index of the row to remove.
        """
        self.J = torch.cat((self.J[:index], self.J[index + 1 :]), dim=0)
        self.h = torch.cat((self.h[:index], self.h[index + 1 :]))

    def _remove_column(self, index: int):
        """
        Remove a given column from the matrix J.

        Parameters
        ----------
        index : int
            Index of the column to remove.
        """
        self.J = torch.cat((self.J[:, :index], self.J[:, index + 1 :]), dim=1)

    def _remove_all_coefficients(self, index: int):
        """
        Remove the row and the column with the same given index
        from both the matrix J and the vector h.

        Parameters
        ----------
        index : int
            Index of the row and column to remove.
        """
        self._remove_row(index)
        self._remove_column(index)

    def _project_coefficients_in_linear_part(self, row_index: int, sign: int):
        """
        Project a row from the matrix J into the vector h.

        Parameters
        ----------
        row_index : int
            Index of the row to project.
        sign : int
            Sign of the spin that acts as a projection coefficient.
        """
        self.h -= sign * self.J[row_index]

    def _project_coefficients_and_delete_row_and_column(
        self, spin_index: int, spin_sign: int
    ):
        """
        Project quadratic coefficients in the vector h and remove
        the associated row and column for a presolved optimized spin.

        Parameters
        ----------
        spin_index : int
            Index of the optimized spin.
        spin_sign : int
            Sign of the optimized spin.
        """
        self._project_coefficients_in_linear_part(spin_index, spin_sign)
        self._remove_all_coefficients(spin_index)

    def _get_optimizable_spins(self) -> torch.Tensor:
        """
        Identify all presolvable spins.

        Returns
        -------
        torch.Tensor
            Boolean tensor that indicates which spins are presolvable.
        """
        return torch.abs(self.h) >= torch.sum(torch.abs(self.J), dim=0)

    def _get_first_optimizable_spin(self) -> Optional[int]:
        """
        Get the index of the first presolvable spin.

        Returns
        -------
        Optional[int]
            Index of the first presolvable spin. If no spin was found,
            None is returned instead.
        """
        optimizable_spins = self._get_optimizable_spins()
        if torch.any(optimizable_spins).item():
            return torch.nonzero(optimizable_spins)[0].item()
        return None

    def _drop_index(self, index: int):
        """
        Remove an index from the shifted indices buffer.

        Parameters
        ----------
        index : int
            Index to drop.
        """
        self.shifted_indices = (
            self.shifted_indices[:index] + self.shifted_indices[index + 1 :]
        )

    def _get_original_index(self, new_index: int) -> int:
        """
        Retrieve the original index of a given index after
        indeces have been shifted because of dropped rows
        and columns.

        Parameters
        ----------
        new_index : int
            Index in the reduced model.

        Returns
        -------
        int
            Original index.
        """
        return self.shifted_indices[new_index]

    def _set_spin_value(self, spin_index: int, spin_sign: int):
        """
        Set the sign of an optimal spin.

        Parameters
        ----------
        spin_index : int
            Index of the optimal spin to set.
        spin_sign : int
            Sign of the optimal spin to set.
        """
        self.optimized_spins[spin_index] = spin_sign

    def _set_first_optimal_spin(self) -> bool:
        """
        Get the first presolvable spin and set its value.
        The coefficients in the tensors are updated and the associated
        row and columns are dropped.

        Returns
        -------
        bool
            Whether a spin has been set or not.
        """
        optimal_spin_index = self._get_first_optimizable_spin()
        if optimal_spin_index is None:
            return False
        sign = -1 if self.h[optimal_spin_index] > 0 else 1
        original_index = self._get_original_index(optimal_spin_index)
        self._set_spin_value(original_index, sign)
        self._project_coefficients_and_delete_row_and_column(optimal_spin_index, sign)
        self._drop_index(optimal_spin_index)
        return True

    def presolve(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Presolves an Ising model.

        Returns
        -------
        optimized_spins : torch.Tensor
            The presolved spins.
        reduced_J : torch.Tensor
            The reduced version of the original J matrix. Only the coefficients
            associated to not presolved spins are remaining.
        reduced_h : torch.Tensor
            The reduced version of the original h vector. Only the coefficients
            associated to not presolved spins are remaining.
        """
        _continue = True
        while _continue:
            optimized_spin = self._set_first_optimal_spin()
            _continue = _continue and optimized_spin and self.J.shape[1] > 0
        return self.optimized_spins.clone(), self.J.clone(), self.h.clone()
