from typing import Tuple, Union

import torch
from tqdm.auto import tqdm


class StopWindow:
    """
    Optimization tool to monitor spins bifurcation and convergence
    for the Simulated Bifurcation (SB) algorithm.
    Allows an early stopping of the iterations and saves computation time.
    """

    def __init__(
        self,
        ising_tensor: torch.Tensor,
        n_agents: int,
        convergence_threshold: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        verbose: bool,
    ) -> None:
        self.ising_tensor = ising_tensor
        self.n_spins = self.ising_tensor.shape[0]
        self.n_agents = n_agents
        self.__init_convergence_threshold(convergence_threshold)
        self.dtype = dtype
        self.device = device
        self.__init_tensors()
        self.__init_energies()
        self.final_spins = self.__init_spins()
        self.progress = self.__init_progress_bar(verbose)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_spins, self.n_agents

    def __init_progress_bar(self, verbose: bool) -> tqdm:
        return tqdm(
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

    def __init_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.n_agents, device=self.device, dtype=dtype)

    def __init_energies(self) -> None:
        self.energies = torch.tensor(
            [float("inf") for _ in range(self.n_agents)], device=self.device
        )

    def __init_tensors(self) -> None:
        self.stability = self.__init_tensor(torch.int16)
        self.newly_bifurcated = self.__init_tensor(torch.bool)
        self.previously_bifurcated = self.__init_tensor(torch.bool)
        self.bifurcated = self.__init_tensor(torch.bool)
        self.stable_agents = self.__init_tensor(torch.bool)

    def __init_spins(self) -> torch.Tensor:
        return torch.zeros(size=self.shape, dtype=self.dtype, device=self.device)

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
        energies = torch.nn.functional.bilinear(
            sampled_spins.t(), sampled_spins.t(), torch.unsqueeze(self.ising_tensor, 0)
        ).reshape(self.n_agents)
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
        return torch.any(self.bifurcated).item()

    def get_bifurcated_spins(self, spins: torch.Tensor) -> torch.Tensor:
        return torch.where(self.bifurcated, self.final_spins, spins)
