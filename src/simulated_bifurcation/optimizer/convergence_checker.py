import torch
from tqdm import tqdm


class ConvergenceChecker(object):
    """
    Optimization tool to monitor agents bifurcation and convergence for the Simulated
    Bifurcation (SB) algorithm. Allows an early stopping of the iterations and saves
    computation time.
    """

    def __init__(
        self,
        convergence_threshold: int,
        ising_tensor: torch.Tensor,
        n_agents: int,
        verbose: bool,
    ) -> None:
        self.__check_convergence_threshold(convergence_threshold)
        self.convergence_threshold = convergence_threshold
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
            disable=not verbose,
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
        """
        Compute the Ising energy (modulo a -2 factor) of the sampled spins.

        Parameters
        ----------
        sampled_spins : torch.Tensor
            Sampled spins provided by the optimizer.

        Returns
        -------
        torch.Tensor
            The energy of each agent.
        """
        return torch.nn.functional.bilinear(
            sampled_spins.t(), sampled_spins.t(), torch.unsqueeze(self.ising_tensor, 0)
        ).reshape(sampled_spins.shape[1])

    def __check_convergence_threshold(self, convergence_threshold: int) -> None:
        """
        Check that the provided convergence threshold is a positive integer.

        Parameters
        ----------
        convergence_threshold : int
            Convergence threshold that defines a convergence criterion for the agents.

        Raises
        ------
        TypeError
            If the convergence threshold is not an integer.
        ValueError
            If the convergence threshold is negative or bigger than 2**15 - 1 (32767).
        """
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

    def update(self, sampled_spins: torch.Tensor) -> torch.Tensor:
        """
        Update the stability streaks and the spins stored in the memory with sampled
        spins from the Simulated Bifurcation optimizer. When an agent converges, it is
        stored in the memory and removed from the optimization process.

        Return a boolean tensor that indicates which agents still have not converged.

        Parameters
        ----------
        sampled_spins : torch.Tensor
            Sampled spins provided by the optimizer.

        Returns
        -------
        torch.Tensor
            The agents that still have not converged (as a boolean tensor).
        """
        self.__update_stability_streaks(sampled_spins)
        self.__update_progressbar(sampled_spins.shape[1])
        return self.__store_converged_spins(sampled_spins)

    def __update_stability_streaks(self, sampled_spins: torch.Tensor):
        """
        Update the stability streaks from the sampled spins provided by the optimizer.

        Parameters
        ----------
        sampled_spins : torch.Tensor
            Sampled spins provided by the optimizer.
        """
        current_agents = self.energies.shape[0]
        energies = self.__compute_energies(sampled_spins)
        stable_agents = torch.eq(energies, self.energies)
        self.energies = energies
        self.stability = torch.where(
            stable_agents,
            self.stability + 1,
            torch.zeros(current_agents, device=self.ising_tensor.device),
        )

    def __store_converged_spins(self, sampled_spins: torch.Tensor) -> torch.Tensor:
        """
        Store the newly converged agents in the memory and updates the utility tensors
        by removing data relative to converged agents.

        Return a boolean tensor that indicates which agents still have not converged.

        Parameters
        ----------
        sampled_spins : torch.Tensor
            Sampled spins provided by the optimizer.

        Returns
        -------
        torch.Tensor
            The agents that still have not converged (as a boolean tensor).
        """
        converged_agents = torch.eq(self.stability, self.convergence_threshold - 1)
        not_converged_agents = torch.logical_not(converged_agents)
        self.stored_spins[:, self.shifted_agents_indices[converged_agents]] = (
            sampled_spins[:, converged_agents]
        )
        self.shifted_agents_indices = self.shifted_agents_indices[not_converged_agents]
        self.energies = self.energies[not_converged_agents]
        self.stability = self.stability[not_converged_agents]
        return not_converged_agents

    def __update_progressbar(self, previous_agents: int):
        """
        Update the progressbar with the number of newly converged agents.

        Parameters
        ----------
        previous_agents : int
            Previous number of agents.
        """
        new_agents = self.energies.shape[0]
        self.progress.update(previous_agents - new_agents)

    def get_stored_spins(self) -> torch.Tensor:
        """
        Return the converged spins stored in the memory.

        Returns
        -------
        torch.Tensor
        """
        return self.stored_spins.clone()
