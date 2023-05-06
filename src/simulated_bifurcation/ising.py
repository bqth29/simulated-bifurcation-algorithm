from typing import List, Tuple, Union
import torch
from numpy import argmin
from .optimizer import Optimizer


class Ising:

    """
    Implementation of an Ising problem to be solved using Simulated
    Bifurcation.

    Solving an Ising problem means searching the spin vector S (with values in
    {-1, 1}) such that, given a symmetric matrix J with zero diagonal and a
    vector h, the following quantity - called Ising energy - is minimal (S is
    then called the ground state):

    `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`


    Attributes
    ----------
    J : torch.Tensor
        spin interactions matrix (must be symmetric with zero diagonal)
    h : torch.Tensor
        magnectic field effect vector
    dimension : int
        number of spins
    ground_state : torch.Tensor
        vector of spins orientation to minimize the Ising energy
    """

    def __init__(
        self, J: torch.Tensor,
        h: Union[torch.Tensor, None] = None,
        dtype: torch.dtype=torch.float32,
        device: str = 'cpu'
    ) -> None:
        """
        Parameters
        ----------
        J : torch.Tensor
            spin interactions matrix (must be symmetric with zero diagonal)
        h : torch.Tensor
            magnectic field effect vector
        """

        if h is None: 
            self.matrix = J.to(device=device, dtype=dtype)
            self.linear_term = False

        elif torch.all(h == 0):
            self.matrix = J.to(device=device, dtype=dtype)
            self.linear_term = False

        else: 
            self.matrix = Ising.attach(J, h, dtype, device)
            self.linear_term = True

        self.dimension = J.shape[0]
        self.computed_spins = None

    def __len__(self) -> int:
        return self.dimension

    def __call__(self, spins: torch.Tensor) -> Union[None, float, List[float]]:

        if spins is None: return None

        elif not isinstance(spins, torch.Tensor):
            raise TypeError(f"Expected a Tensor but got {type(spins)}.")

        elif torch.any(torch.abs(spins) != 1):
            raise ValueError('Spins must be either 1 or -1.')

        elif spins.shape in [(self.dimension,), (self.dimension, 1)]:
            spins = spins.reshape((-1, 1))
            J, h = self.J, self.h.reshape((-1, 1))
            energy = -.5 * spins.t() @ J @ spins + spins.t() @ h
            return energy.item()

        elif spins.shape[0] == self.dimension:
            J, h = self.J, self.h.reshape((-1, 1))
            energies = torch.einsum('ij, ji -> i', spins.t(), -.5 * J @ spins + h)
            return energies.tolist()

        else:
            raise ValueError(f"Expected {self.dimension} rows, got {spins.shape[0]}.")


    @classmethod
    def attach(
        cls, J: torch.Tensor, h: torch.Tensor,
        dtype: torch.dtype=torch.float32,
        device: str='cpu'
    ) -> torch.Tensor:

        dimension = J.shape[0]
        matrix = torch.zeros(
            (dimension + 1, dimension + 1),
            dtype=dtype, device=device
        )

        matrix[:dimension, :dimension] = J
        matrix[:dimension, dimension] = - h.reshape(-1,)
        matrix[dimension, :dimension] = - h.reshape(-1,)

        return matrix

    @classmethod
    def detach(
        cls, matrix: torch.Tensor,
        dtype: torch.dtype=torch.float32,
        device: str='cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dimension = matrix.shape[0] - 1

        J = matrix[:dimension, :dimension].to(
            dtype=dtype, device=device)
        h = - matrix[:dimension, dimension].to(
            dtype=dtype, device=device)

        return J, h

    @classmethod
    def remove_diagonal(cls, matrix: torch.Tensor) -> torch.Tensor:
        return matrix - torch.diag(torch.diag(matrix))

    @property
    def dtype(self) -> torch.dtype: return self.matrix.dtype

    @property
    def device(self) -> torch.device: return self.matrix.device

    @property
    def shape(self) -> Tuple[int, int]: return self.matrix.shape

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
        time_step: float = .1,
        convergence_threshold: int = 50,
        sampling_period: int = 50,
        max_steps: int = 10000,
        agents: int = 128,
        pressure_slope: float = .01,
        gerschgorin: bool = False,
        use_window: bool = True,
        ballistic: bool = False,
        heat_parameter: float = None,
        verbose: bool = True
    ):
        """
        Computes an approximated solution of the Ising problem using the
        Simulated Bifurcation algorithm. The ground state in modified in place.
        It should correspond to a local minimum for the Ising energy function.

        The Simulated Bifurcation (SB) algorithm mimics Hamiltonian dynamics to
        make spins evolve throughout time. It uses a symplectic Euler scheme
        for this purpose.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually slower but more accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually faster but less accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the Euler scheme, a number of maximum steps
        needs to be specified. However a refined way to stop is also possible
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
        spin vector (energy-wise) is kept and used as the new Ising model'
        ground state.

        Parameters
        ----------

        - Euler scheme parameters

        time_step : float, optional
            step size for the time discretization (default is 0.01)
        symplectic_parameter : int | 'inf', optional
            symplectic parameter for the Euler's scheme (default is 2)
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 35)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 60000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 20)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)

        - Quantum parameters

        detuning_frequency : float, optional
            detuning frequency of the Hamiltonian (default is 1.0)
        pressure_slope : float, optional
            pumping pressure's linear slope allowing adiabatic evolution
            (default is 0.01)
        final_pressure : float | None, optional
            pumping pressure's maximum value; if None, no maximum value is set
            (default is None)
        xi0 : float | 'gerschgorin' | None, optional
            weighting coefficient in the Hamiltonian; if None it will be
            computed based on the J matrix (default is None)
        heat_parameter : float, optional
            heat parameter for the heated SB algorithm (default is 0.06)

        - Others

        verbose : bool, optional
            whether to display evolution information or not (default is True)

        See Also
        --------

        For more information on the Hamiltonian equations, check
        `SymplecticEulerScheme`.

        Notes
        -----

        For low dimensions, see the `comprehensive_search` method function
        instead that will always find the true optimal ground state.
        """
        optimizer = Optimizer(time_step, convergence_threshold,
                      sampling_period, max_steps, agents,
                      pressure_slope, gerschgorin, ballistic, 
                      heat_parameter, verbose)
        
        matrix = Ising.remove_diagonal(self.matrix)
        spins = optimizer.symplectic_update(matrix, use_window)
        
        if self.linear_term: self.computed_spins = spins[-1] * spins[:-1, :]
        else: self.computed_spins = spins


