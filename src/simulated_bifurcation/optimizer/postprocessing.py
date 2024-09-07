import torch


class Postprocessing:
    """
    Utilility class to reconstrcut the spin agents from presolved spins
    on the original Ising model and spins optimized by the Simulated
    Bifurcation algorithm on the reduced Ising model.
    """

    @staticmethod
    def reconstruct_spins(
        optimized_spins: torch.Tensor, pre_solved_spins: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct the spin vectors by merging the presolved spins and
        the spins optimized by the Simulated Bifurcation algorithm on the
        reduced Ising model.

        Parameters
        ----------
        optimized_spins : torch.Tensor
            Spins returned by the Simulated Bifurcation algorithm
            that correspond to the presolved reduced Ising model.
        pre_solved_spins : torch.Tensor
            Presolved spins of the original Ising model.

        Returns
        -------
        torch.Tensor
            Reconstructed spins taking in account both the SB-optimized
            spins and the presolved spins.
        """
        original_dimension = pre_solved_spins.shape[0]
        agents = optimized_spins.shape[1]
        reconstructed_spins = torch.zeros(original_dimension, agents, dtype=torch.int32)
        reconstructed_spins[pre_solved_spins == 0] = optimized_spins.clone().to(
            dtype=torch.int32
        )
        reconstructed_spins[torch.abs(pre_solved_spins) == 1] = (
            pre_solved_spins[torch.abs(pre_solved_spins) == 1]
            .repeat(agents, 1)
            .t()
            .to(dtype=torch.int32)
        )
        return reconstructed_spins
