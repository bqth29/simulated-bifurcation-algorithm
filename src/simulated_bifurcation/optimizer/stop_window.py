from typing import Tuple, Union

import torch
from tqdm import tqdm


class StopWindow:

    """
    Optimization tool to monitor spins bifurcation and convergence
    for the Simulated Bifurcation (SB) algorithm.
    Allows an early stopping of the iterations and saves computation time.
    """

    def __init__(
        self,
        convergence_threshold: int,
        ising_tensor: torch.Tensor,
        n_agents: int,
        verbose: bool,
    ) -> None:
        self.__init_convergence_threshold(convergence_threshold)
        self.verbose = verbose
        self.ising_tensor = ising_tensor
        self.stability = torch.zeros(
            n_agents, dtype=torch.int16, device=ising_tensor.device
        )
        self.energies = torch.tensor(
            [float("inf") for _ in range(n_agents)],
            dtype=ising_tensor.dtype,
            device=ising_tensor.device,
        )
        self.progress = tqdm(
            total=n_agents,
            desc="🏁 Bifurcated agents",
            disable=not self.verbose,
            smoothing=0,
            unit=" agents",
        )
        self.stored_spins = torch.zeros(
            ising_tensor.shape[0],
            n_agents,
            dtype=ising_tensor.dtype,
            device=ising_tensor.device,
        )
        self.shifted_agents_indices = torch.tensor(
            list(range(n_agents)), device=ising_tensor.device
        )

    def __compute_energies(self, sampled_spins: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.bilinear(
            sampled_spins.t(), sampled_spins.t(), torch.unsqueeze(self.ising_tensor, 0)
        ).reshape(sampled_spins.shape[1])

    def __init_convergence_threshold(self, convergence_threshold: int) -> None:
        if not isinstance(convergence_threshold, int):
            raise TypeError(
                "convergence_threshold should be an integer, "
                f"received {convergence_threshold}."
            )
        if convergence_threshold <= 0:
            raise ValueError(
                "convergence_threshold should be a positive integer, "
                f"received {convergence_threshold}."
            )
        if convergence_threshold > torch.iinfo(torch.int16).max:
            raise ValueError(
                "convergence_threshold should be less than or equal to "
                f"{torch.iinfo(torch.int16).max}, received {convergence_threshold}."
            )
        self.convergence_threshold = convergence_threshold

    def update(self, sampled_spins: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        sampled_spins : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            The agents that still have not converged.
        """
        current_agents = self.energies.shape[0]
        energies = self.__compute_energies(sampled_spins)
        stable_agents = torch.eq(energies, self.energies)
        self.energies = energies
        self.stability = torch.where(
            stable_agents, self.stability + 1, torch.zeros(current_agents)
        )

        converged_agents = torch.eq(self.stability, self.convergence_threshold - 1)
        not_converged_agents = torch.logical_not(converged_agents)

        self.stored_spins[
            :, self.shifted_agents_indices[converged_agents]
        ] = sampled_spins[:, converged_agents]

        self.shifted_agents_indices = self.shifted_agents_indices[not_converged_agents]
        self.energies = self.energies[not_converged_agents]
        self.stability = self.stability[not_converged_agents]
        new_agents = self.energies.shape[0]
        self.progress.update(current_agents - new_agents)
        return not_converged_agents

    def get_stored_spins(self) -> torch.Tensor:
        """
        Returns the converged spins stored in the window.

        Returns
        -------
        torch.Tensor
        """
        return self.stored_spins.clone()
