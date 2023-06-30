from typing import List, Tuple, Union
import torch
from numpy import argmin, ndarray
from .optimizer import SimulatedBifurcationOptimizer


class Ising:

    """
    Implementation of the Ising model.

    Solving an Ising problem means searching the spin vector S (with values in
    {-1, 1}) such that, given a matrix J with zero diagonal and a
    vector h, the following quantity - called Ising energy - is minimal (S is
    then called the ground state): `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`
    """

    def __init__(
        self, J: Union[torch.Tensor, ndarray],
        h: Union[torch.Tensor, ndarray, None] = None,
        dtype: torch.dtype=torch.float32,
        device: str = 'cpu'
    ) -> None:
        if isinstance(J, torch.Tensor):
            self.__init_from_tensor(J, h, dtype, device)
        else:
            self.__init_from_array(J, h, dtype, device)
        self.dimension = J.shape[0]
        self.computed_spins = None

    def __len__(self) -> int:
        return self.dimension

    def __call__(self, spins: torch.Tensor) -> Union[None, float, List[float]]:
        if spins is None: return None
        if isinstance(spins, ndarray):
            spins = torch.from_numpy(spins).to(dtype=self.dtype, device=self.device)
        if not isinstance(spins, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(spins)}.")
        if torch.any(torch.abs(spins) != 1):
            raise ValueError('Spins must be either 1 or -1.')
        if spins.shape in [(self.dimension,), (self.dimension, 1), (1, self.dimension)]:
            spins = spins.reshape((-1, 1))
            J, h = self.J, self.h.reshape((-1, 1))
            energy = -.5 * spins.t() @ J @ spins + spins.t() @ h
            return energy.item()
        if spins.shape[0] == self.dimension:
            J, h = self.J, self.h.reshape((-1, 1))
            energies = torch.einsum('ij, ji -> i', spins.t(), -.5 * J @ spins + h)
            return energies.tolist()
        else:
            raise ValueError(f"Expected {self.dimension} rows, got {spins.shape[0]}.")
        
    def __init_from_tensor(self, J: torch.Tensor, h: Union[torch.Tensor, None],
                          dtype: torch.dtype, device: str): 
        if h is None: 
            self.matrix = J.to(device=device, dtype=dtype)
            self.linear_term = False
        elif torch.all(h == 0):
            self.matrix = J.to(device=device, dtype=dtype)
            self.linear_term = False
        else: 
            self.matrix = Ising.attach(J, h, dtype, device)
            self.linear_term = True

    def __init_from_array(self, J: ndarray, h: Union[ndarray, None],
                          dtype: torch.dtype, device: str):
        self.__init_from_tensor(
            torch.from_numpy(J),
            None if h is None else torch.from_numpy(h),
            dtype, device
        )

    @staticmethod
    def attach(
        J: torch.Tensor, h: torch.Tensor,
        dtype: torch.dtype=torch.float32,
        device: str='cpu'
    ) -> torch.Tensor:
        """
        Gathers the matrix and the vector of the Ising model
        into a single matrix that can be processed by the
        Simulated Bifurcation (SB) algorithm.
        """
        dimension = J.shape[0]
        matrix = torch.zeros(
            (dimension + 1, dimension + 1),
            dtype=dtype, device=device
        )
        matrix[:dimension, :dimension] = J
        matrix[:dimension, dimension] = - h.reshape(-1,)
        matrix[dimension, :dimension] = - h.reshape(-1,)
        return matrix

    @staticmethod
    def detach(
        matrix: torch.Tensor,
        dtype: torch.dtype=torch.float32,
        device: str='cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the Simulated Bifurcation (SB) algorithm's matrix
        into the initial ising model's matrix and vector.
        """
        dimension = matrix.shape[0] - 1
        J = matrix[:dimension, :dimension].to(
            dtype=dtype, device=device)
        h = - matrix[:dimension, dimension].to(
            dtype=dtype, device=device)
        return J, h

    @staticmethod
    def remove_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        return matrix - torch.diag(torch.diag(matrix))
    
    @staticmethod
    def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return .5 * (matrix + matrix.t())
    
    @staticmethod
    def format_matrix(matrix: torch.Tensor) -> torch.Tensor:
        """
        Transform the Ising's model matrix to an equivalent form
        that is more suitable to be processed by the 
        Simulated Bifurcation (SB) algorithm:
        1. The matrix is averaged with its transpose (symmetrization);
        2. The matrix's diagonal is removed.
        """
        return Ising.remove_diagonal(Ising.symmetrize(matrix))

    @property
    def dtype(self) -> torch.dtype: return self.matrix.dtype

    @property
    def device(self) -> torch.device: return self.matrix.device

    @property
    def ground_state(self) -> Union[torch.Tensor, None]:
        if self.computed_spins is None: return None
        else: return self.min(self.computed_spins)

    @property
    def energy(self) -> Union[float, None]: return self(self.ground_state)

    @property
    def J(self) -> torch.Tensor: 
        if self.linear_term:
            return Ising.detach(
                self.matrix,
                self.dtype,
                self.device
            )[0]
        else: return self.matrix

    @property
    def h(self) -> torch.Tensor: 
        if self.linear_term:
            return Ising.detach(
                self.matrix,
                self.dtype,
                self.device
            )[1]
        else:
            return torch.zeros(
                self.dimension,
                dtype=self.dtype,
                device=self.device
            )

    def min(self, spins: torch.Tensor) -> torch.Tensor:
        """
        Returns the spin vector with the lowest Ising energy.
        """
        energies = self(spins)
        best_energy = argmin(energies)
        return spins[:, best_energy]

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
        optimizer = SimulatedBifurcationOptimizer(convergence_threshold,
                      sampling_period, max_steps, agents,
                      ballistic, heat, verbose)
        matrix = Ising.format_matrix(self.matrix)
        spins = optimizer.run_integrator(matrix, use_window)
        self.computed_spins = spins[-1] * spins[:-1, :] if self.linear_term else spins
