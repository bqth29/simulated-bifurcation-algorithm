import torch


class Postprocessing:
    @staticmethod
    def reconstruct_spins(
        optimized_spins: torch.Tensor, pre_solved_spins: torch.Tensor
    ) -> torch.Tensor:
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
