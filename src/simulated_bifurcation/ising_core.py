from typing import Optional, Union

import torch
from numpy import ndarray

from .optimizer import OptimizerMode, SimulatedBifurcationOptimizer


class IsingCore:

    """
    Implementation of the Ising model.

    Solving an Ising problem means searching the spin vector S (with values in
    {-1, 1}) such that, given a matrix J with zero diagonal and a
    vector h, the following quantity - called Ising energy - is minimal (S is
    then called the ground state): `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`
    """

    def __init__(
        self,
        J: Union[torch.Tensor, ndarray],
        h: Union[torch.Tensor, ndarray, None] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.dimension = J.shape[0]
        if isinstance(J, ndarray):
            J = torch.from_numpy(J)
        if isinstance(h, ndarray):
            h = torch.from_numpy(h)
        self.__init_from_tensor(J, h, dtype, device)
        self.computed_spins = None

    def __len__(self) -> int:
        return self.dimension

    def __neg__(self):
        return self.__class__(-self.J, -self.h, self.dtype, self.device)

    def __init_from_tensor(
        self,
        J: torch.Tensor,
        h: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: str,
    ):
        null_vector = torch.zeros(self.dimension).to(device=device, dtype=dtype)
        if h is None:
            self.J = J.to(device=device, dtype=dtype)
            self.h = null_vector
            self.linear_term = False
        elif torch.equal(
            h.reshape(self.dimension).to(device=device, dtype=dtype), null_vector
        ):
            self.J = J.to(device=device, dtype=dtype)
            self.h = null_vector
            self.linear_term = False
        else:
            self.J = J.to(device=device, dtype=dtype)
            self.h = h.reshape(self.dimension).to(device=device, dtype=dtype)
            self.linear_term = True

    def clip_vector_to_tensor(self) -> torch.Tensor:
        """
        Gathers the matrix and the vector of the Ising model
        into a single matrix that can be processed by the
        Simulated Bifurcation (SB) algorithm.
        """
        tensor = torch.zeros(
            (self.dimension + 1, self.dimension + 1),
            dtype=self.dtype,
            device=self.device,
        )
        tensor[: self.dimension, : self.dimension] = self.J
        tensor[: self.dimension, self.dimension] = -self.h
        tensor[self.dimension, : self.dimension] = -self.h
        return tensor

    @staticmethod
    def remove_diagonal(tensor: torch.Tensor) -> torch.Tensor:
        return tensor - torch.diag(torch.diag(tensor))

    @staticmethod
    def symmetrize(tensor: torch.Tensor) -> torch.Tensor:
        return 0.5 * (tensor + tensor.t())

    def as_simulated_bifurcation_tensor(self) -> torch.Tensor:
        tensor = self.remove_diagonal(self.symmetrize(self.J))
        if self.linear_term:
            sb_tensor = self.clip_vector_to_tensor()
        else:
            sb_tensor = tensor
        return sb_tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.J.dtype

    @property
    def device(self) -> torch.device:
        return self.J.device

    def optimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
    ):
        """
        Computes a local minimum of the Ising problem using the
        Simulated Bifurcation (SB) algorithm.
        The ground state in modified in place.

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually faster but less accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually slower but more accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However, a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps to explore the solution space
        and increases the probability of finding a better solution, though it
        also slightly increases the computation time. In the end, only the best
        spin vector (energy-wise) is kept and used as the new Ising model's
        ground state.

        Parameters
        ----------
        *
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
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        """
        optimizer = SimulatedBifurcationOptimizer(
            agents,
            max_steps,
            OptimizerMode.BALLISTIC if ballistic else OptimizerMode.DISCRETE,
            heated,
            verbose,
            sampling_period,
            convergence_threshold,
        )
        tensor = self.as_simulated_bifurcation_tensor()
        spins = optimizer.run_integrator(tensor, use_window)
        if self.linear_term:
            self.computed_spins = spins[-1] * spins[:-1]
        else:
            self.computed_spins = spins
