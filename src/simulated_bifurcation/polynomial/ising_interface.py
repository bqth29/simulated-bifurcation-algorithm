from ..ising import Ising
from .polynomial import Polynomial
from abc import ABC, abstractmethod
from typing import final, List, Union
import torch
import numpy as np


class IsingInterface(Polynomial):

    """
    Abstract class to derive optimization problems from Ising models.
    """

    def __init__(self, matrix: torch.Tensor, vector: torch.Tensor, constant: Union[int, float], accepted_values: List[int], dtype: torch.dtype, device: str) -> None:
        super().__init__(matrix, vector, constant, accepted_values, dtype, device)
        self.sb_result = None

    @abstractmethod
    def to_ising(self) -> Ising:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.
        """
        raise NotImplementedError

    @abstractmethod
    def from_ising(self, ising: Ising) -> torch.Tensor:
        """
        Retrieves information from the optimized equivalent Ising model.
        Returns the best found vector.

        Parameters
        ----------
        ising : Ising
            equivalent Ising model of the problem
        """
        raise NotImplementedError

    @final
    def optimize(
        self,
        convergence_threshold: int = 50,
        sampling_period: int = 50,
        max_steps: int = 10000,
        agents: int = 128,
        use_window: bool = True,
        ballistic: bool = False,
        heat: bool = False,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Computes a local extremum of the model by optimizing
        the equivalent Ising model using the Simulated
        Simulated Bifurcation (SB) algorithm. 

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually slower but more accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually faster but less accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps exploring the solution space
        and increases the probability of finding a better solution, though it
        also slightly increases the computation time. In the end, only the best
        spin vector (energy-wise) is kept and used as the new Ising model's
        ground state.

        Parameters
        ----------
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 50)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 10000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 128)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heat : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        """
        ising_equivalent = self.to_ising()
        ising_equivalent.optimize(
            convergence_threshold, sampling_period,
            max_steps, agents, use_window, ballistic, heat, verbose
        )
        self.sb_result = self.from_ising(ising_equivalent)
        return self.sb_result
