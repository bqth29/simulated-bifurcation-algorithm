from typing import Tuple, Union

import torch
from tqdm import tqdm


class StopWindow:

    """
    Optimization tool to monitor spins bifurcation and convergence
    for the Simulated Bifurcation (SB) algorithm.
    Allows an early stopping of the iterations and saves computation time.
    """

    def __init__(self, convergence_threshold: int, verbose: bool) -> None:
        self.__init_convergence_threshold(convergence_threshold)
        self.verbose = verbose
        self.ising_tensor = None
        self.n_agents = None
        self.stability = None
        self.newly_bifurcated = None
        self.previously_bifurcated = None
        self.bifurcated = None
        self.stable_agents = None
        self.energies = None
        self.progress = None
        self.current_spins = None
        self.final_spins = None
        self.shifted_agents_index = None

    def reset(self, ising_tensor: torch.Tensor, n_agents: int):
        self.ising_tensor = ising_tensor
        self.n_agents = n_agents
        self.__init_tensors(ising_tensor.device)
        self.__init_energies(ising_tensor.dtype, ising_tensor.device)
        self.__init_progress_bar(self.verbose)
        self.__init_current_and_final_spins(
            ising_tensor.shape[0], n_agents, ising_tensor.dtype, ising_tensor.device
        )
        self.shifted_agents_index = list(range(n_agents))

    def __compute_energies(self, sampled_spins: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.bilinear(
            sampled_spins.t(), sampled_spins.t(), torch.unsqueeze(self.ising_tensor, 0)
        ).reshape(self.n_agents)

    def __init_progress_bar(self, verbose: bool) -> None:
        self.progress = tqdm(
            total=self.n_agents,
            desc="🏁 Bifurcated agents",
            disable=not verbose,
            smoothing=0,
            unit=" agents",
        )

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

    def __init_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_agents, device=device, dtype=dtype)

    def __init_tensors(self, device: torch.device) -> None:
        self.stability = self.__init_tensor(torch.int16, device)
        self.newly_bifurcated = self.__init_tensor(torch.bool, device)
        self.previously_bifurcated = self.__init_tensor(torch.bool, device)
        self.bifurcated = self.__init_tensor(torch.bool, device)
        self.stable_agents = self.__init_tensor(torch.bool, device)

    def __init_energies(self, dtype: torch.dtype, device: torch.device) -> None:
        self.energies = torch.tensor(
            [float("inf") for _ in range(self.n_agents)], dtype=dtype, device=device
        )

    def __init_spins(
        self, n_spins: int, n_agents: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(n_spins, n_agents, dtype=dtype, device=device)

    def __init_current_and_final_spins(
        self, n_spins: int, n_agents: int, dtype: torch.dtype, device: torch.device
    ):
        self.current_spins = self.__init_spins(n_spins, n_agents, dtype, device)
        self.final_spins = self.__init_spins(n_spins, n_agents, dtype, device)

    def __update_final_spins(self, sampled_spins) -> None:
        self.final_spins[:, self.newly_bifurcated] = sampled_spins[
            :, self.newly_bifurcated
        ]

    def __set_previously_bifurcated_spins(self) -> None:
        self.previously_bifurcated = torch.clone(self.bifurcated)

    def __set_newly_bifurcated_spins(self) -> None:
        torch.logical_xor(
            self.bifurcated, self.previously_bifurcated, out=self.newly_bifurcated
        )

    def __update_bifurcated_spins(self) -> None:
        torch.eq(self.stability, self.convergence_threshold - 1, out=self.bifurcated)

    def __update_stability_streak(self) -> None:
        self.stability[torch.logical_and(self.stable_agents, self.not_bifurcated)] += 1
        self.stability[torch.logical_and(self.changed_agents, self.not_bifurcated)] = 0

    @property
    def changed_agents(self) -> torch.Tensor:
        return torch.logical_not(self.stable_agents)

    @property
    def not_bifurcated(self) -> torch.Tensor:
        return torch.logical_not(self.bifurcated)

    def __compare_energies(self, sampled_spins: torch.Tensor) -> None:
        energies = self.__compute_energies(sampled_spins)
        torch.eq(
            energies,
            self.energies,
            out=self.stable_agents,
        )
        self.energies = energies

    def __get_number_newly_bifurcated_agents(self) -> int:
        return torch.count_nonzero(self.newly_bifurcated).item()

    def update(self, sampled_spins: torch.Tensor):
        self.__compare_energies(sampled_spins)
        self.__update_stability_streak()
        self.__update_bifurcated_spins()
        self.__set_newly_bifurcated_spins()
        self.__set_previously_bifurcated_spins()
        self.__update_final_spins(sampled_spins)
        self.progress.update(self.__get_number_newly_bifurcated_agents())

    def must_continue(self) -> bool:
        return torch.any(
            torch.lt(self.stability, self.convergence_threshold - 1)
        ).item()

    def has_bifurcated_spins(self) -> bool:
        return torch.any(torch.not_equal(self.final_spins, 0)).item()

    def get_final_spins(self, spins: torch.Tensor) -> torch.Tensor:
        """
        Returns the final spins of the window. If an agent did not converge,
        the spins provided in input are returned instead.

        Parameters
        ----------
        spins : torch.Tensor
            Spins coming from the optimizer.

        Returns
        -------
        torch.Tensor
            Final spins.
        """
        return torch.where(self.bifurcated, self.final_spins, spins)
