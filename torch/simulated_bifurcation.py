from abc import ABC, abstractmethod
import itertools as it
import textwrap
from time import time
from typing import final

import torch
from tqdm import tqdm
from numpy import minimum


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

    def __init__(self, J: torch.Tensor, h: torch.Tensor,
                dtype: torch.dtype=torch.float32,
                device: torch.device=torch.device('cpu')) -> None:
        """
        Parameters
        ----------
        J : torch.Tensor
            spin interactions matrix (must be symmetric with zero diagonal)
        h : torch.Tensor
            magnectic field effect vector
        """
        self.J = J.to(dtype).to(device)
        self.null_diag_J = self.J - torch.diag(torch.diag(self.J))
        self.h = h.to(dtype).to(device)

        self.dimension = J.shape[0]
        self.ground_state = None

        self.dtype = dtype
        self.device = device

    def __str__(self) -> str:
        if self.ground_state is None:
            return 'Non-optimized Ising model'
        str_ground_state = ''.join(map(lambda s: '+' if s > 0 else '-',
                                       self.ground_state.reshape(-1,)))
        message = textwrap.dedent(f"""
        Optimized Ising model
        - Spins: {self.dimension}
        - Best ground state: {str_ground_state}
        - Energy: {self.energy}
        """)
        return message

    def __len__(self) -> int:
        return self.dimension

    @property
    def energy(self) -> float:
        """
        Ising energy of the model.
        """
        if self.ground_state is None:
            return None
        energy = -0.5 * self.ground_state.t() @ self.J @ self.ground_state + \
            self.ground_state.t() @ self.h
        return energy[0][0].item()

    def optimize(
        self,
        time_step: float = .01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 60,
        sampling_period: int = 35,
        max_steps: int = 60000,
        agents: int = 20,
        detuning_frequency: float = 1.,
        pressure_slope: float = .01,
        final_pressure: float = 1.,
        xi0: float = None,
        heat_parameter: float = 0.06,
        use_window: bool = True,
        ballistic: bool = False,
        heated: bool = True,
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
        if ballistic and heated:
            algo = BallisticHeatedSymplecticEulerScheme
        elif ballistic and not heated:
            algo = BallisticSymplecticEulerScheme
        elif not ballistic and heated:
            algo = DiscreteHeatedSymplecticEulerScheme
        else:
            algo = DiscreteSymplecticEulerScheme
        solver = algo(time_step, symplectic_parameter, convergence_threshold,
                      sampling_period, max_steps, agents, detuning_frequency,
                      pressure_slope, final_pressure, xi0, heat_parameter,
                      verbose)
        self.ground_state = solver.iterate(self, use_window)


class SBModel(ABC):
    """
    An abstract class to adapt optimization problems as Ising problems.
    """

    @abstractmethod
    def __to_Ising__(self) -> Ising:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.
        Thus, there may be no scientific signification of this
        equivalent model.

        Returns
        -------
        Ising
        """
        raise NotImplementedError

    @abstractmethod
    def __from_Ising__(self, ising: Ising) -> None:
        """
        Retrieves information from the optimized equivalent Ising model.
        Modifies the object's attributes in place.

        Parameters
        ----------
        ising : Ising
            equivalent Ising model of the problem
        """
        raise NotImplementedError

    @final
    def optimize(
        self,
        time_step: float = .01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 60,
        sampling_period: int = 35,
        max_steps: int = 60000,
        agents: int = 20,
        detuning_frequency: float = 1.,
        pressure_slope: float = .01,
        final_pressure: float = 1.,
        xi0: float = None,
        heat_parameter: float = 0.06,
        use_window: bool = True,
        ballistic: bool = False,
        heated: bool = True,
        verbose: bool = True
    ) -> None:
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

        For more information on the Hamiltonian parameters, check
        `SymplecticEulerScheme`.

        Notes
        -----

        For low dimensions, see the `comprehensive_search` method function
        instead that will always find the true optimal ground state.
        """
        ising_equivalent = self.__to_Ising__()
        ising_equivalent.optimize(
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period,
            max_steps,
            agents,
            detuning_frequency,
            pressure_slope,
            final_pressure,
            xi0,
            heat_parameter,
            use_window,
            ballistic,
            heated,
            verbose
        )
        self.__from_Ising__(ising_equivalent)


class SymplecticEulerScheme(ABC):
    """
    An abstract class to implement a Symplectic Euler Scheme to perform the
    Simulated Bifurcation (SB) algorithm.

    Attributes
    ---------
    time_step : float
        step size for the time discretization
    symplectic_parameter : int | 'inf'
        symplectic parameter for the Euler's scheme
    symplectic_time_step : float
        symplectic step size for the time discretization
    agents : int
        number of vectors to make evolve at the same time
    convergence_threshold : int
        number of consecutive identical spin sampling considered as a proof of
        convergence
    sampling_period : int
        number of time steps between two spin sampling
    max_steps : int
        number of time steps after which the algorithm will stop inevitably
    detuning_frequency : float
        detuning frequency of the Hamiltonian
    pressure : function
        pumping pressure's linear slope allowing adiabatic evolution
    field_coefficient : function
        evolutive coefficient before the magnetic field term in the Hamiltonian
        allowing adiabatic evolution
    xi0 : float | 'gerschgorin' | None
        weighting coefficient in the Hamiltonian
    heat_parameter : float
        heat parameter for the heated SB algorithm
    X : torch.Tensor | None
        particles' position vectors
    Y : torch.Tensor | None
        particles' pulsation vectors
    dimension : int | None
        dimension of the Ising problem to solve
    current_spins : torch.Tensor | None
        current value of the spins
    stability : torch.Tensor | None
        vector gathering, for each agent, the stability of its spins throughout
        time (used by the window)
    run : bool
        while this attribute is True, the scheme will be iterated; set to False
        when the stopping criterion is met
    step : int
        current number of iterations
    time : float | None
        time required for the spins to bifurcate (i.e. simulation time)
    agents_progress : tqdm.tqdm
        progress bar displayed in the console to inform on the state of the
        simulation
    iterations_progress : tqdm.tqdm
        progress bar displayed in the console to inform on the state of the
        simulation
    """

    def __init__(
        self,
        time_step: float = .01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 60,
        sampling_period: int = 35,
        max_steps: int = 60000,
        agents: int = 20,
        detuning_frequency: float = 1.,
        pressure_slope: float = .01,
        final_pressure: float = None,
        xi0: float = None,
        heat_parameter: float = 0.06,
        verbose: bool = True
    ) -> None:
        """
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
        """

        self.agents_progress = tqdm(total=agents, desc='Bifurcated agents',
                                    disable=not verbose, smoothing=0)
        self.iterations_progress = tqdm(total=max_steps, desc='Iterations',
                                        disable=not verbose, smoothing=0.1,
                                        mininterval=0.5)

        # Simulation parameters
        self.time_step = time_step
        self.symplectic_parameter = symplectic_parameter
        if symplectic_parameter != 'inf':
            self.symplectic_time_step = time_step / symplectic_parameter
        self.agents = agents

        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps

        # Quantum parameters
        self.detuning_frequency = detuning_frequency
        self.heat_parameter = heat_parameter
        self.xi0 = xi0

        if final_pressure is None:
            self.pressure = lambda t: pressure_slope * t
        else:
            self.pressure = lambda t: minimum(
                pressure_slope * t, final_pressure)

        # self.field_coefficient = lambda t: np.sqrt(
        #     self.pressure(t) - self.detuning_frequency * np.tanh(
        #         self.pressure(t) / self.detuning_frequency
        #     )
        # )

        self.field_coefficient = self.pressure

        # Evolutive parameters
        self.X, self.Y = None, None
        self.dimension = None
        self.current_spins = None
        self.final_spins = None
        self.stability = None
        self.bifurcated = None
        self.previously_bifurcated = None
        self.new_bifurcated = None
        self.equal = None
        self.run = True
        self.step = 0
        self.time = 0

    @final
    def confine(self) -> None:
        """
        Confine the particles' position in the range [-1, 1], i.e. if a `x > 1`
        or `x < -1`, `x` is replaced by `sign(x)` and the corresponding
        pulsation `y` is set to 0.
        """
        torch.clip(self.X, -1., 1., out=self.X)
        self.Y[torch.abs(self.X) == 1.] = 0

    @final
    def update_window(self) -> None:
        """
        Sample the current spins and compare them to the previous ones.
        Modify the stability vector in place.
        """
        torch.eq(torch.einsum('ik, ik -> k', self.current_spins, torch.sign(self.X)),
                 self.dimension, out=self.equal)
        not_bifurcated = torch.logical_not(self.bifurcated)
        not_equal = torch.logical_not(self.equal)
        self.stability[torch.logical_and(self.equal, not_bifurcated)] += 1
        self.stability[torch.logical_and(not_equal, not_bifurcated)] = 0

        torch.eq(self.stability, self.convergence_threshold - 1,
                 out=self.bifurcated)

        torch.logical_xor(self.bifurcated, self.previously_bifurcated,
                       out=self.new_bifurcated)

        self.previously_bifurcated = self.bifurcated * 1.

        self.final_spins[:, self.new_bifurcated] = torch.sign(
            self.X[:, self.new_bifurcated])

        torch.sign(self.X, out=self.current_spins)

        self.agents_progress.update(self.new_bifurcated.sum().item())

    @final
    def reset(self, ising: Ising) -> None:
        """
        Reset the simulation parameters.

        Parameters
        ----------
        ising : Ising
            the Ising model to solve
        """
        self.dimension = ising.dimension
        self.X = 2 * torch.rand(size=(self.dimension, self.agents), 
                device = ising.device, dtype=ising.dtype) - 1
        self.Y = 2 * torch.rand(size=(self.dimension, self.agents),
                device = ising.device, dtype=ising.dtype) - 1

        self.current_spins = torch.zeros((self.dimension, self.agents),
                device = ising.device, dtype=ising.dtype)
        self.final_spins = torch.zeros((self.dimension, self.agents),
                device = ising.device, dtype=ising.dtype)

        self.stability = torch.zeros(self.agents,
                device = ising.device, dtype=ising.dtype)
        self.new_bifurcated = torch.zeros(self.agents, dtype=bool,
                device = ising.device)
        self.previously_bifurcated = torch.zeros(self.agents, dtype=bool,
                device = ising.device)
        self.bifurcated = torch.zeros(self.agents, dtype=bool,
                device = ising.device)
        self.equal = torch.zeros(self.agents, dtype=bool,
                device = ising.device)

        self.run = True

        self.step = 0
        self.time = 0

        if self.xi0 is None:
            self.xi0 = 0.7 * self.detuning_frequency / \
                (torch.std(ising.null_diag_J) * (ising.dimension)**(1/2))
        elif self.xi0 == 'gerschgorin':
            self.xi0 = self.detuning_frequency / torch.max(
                torch.sum(torch.abs(ising.null_diag_J), axis=1))
        else:
            pass

    @final
    def step_update(self) -> None:
        """
        Increments the current step by 1.
        """
        self.step += 1
        self.iterations_progress.update()

    @final
    def symplectic_update(self) -> None:
        """
        Update the particle vectors with the symplectic part of the Hamiltonian
        equations.

        Parameters
        ----------
        ising : Ising
            the Ising model to solve
        """
        pressure = self.pressure(self.time_step * self.step)

        if self.symplectic_parameter != 'inf':
            for _ in range(self.symplectic_parameter):
                torch.add(self.Y, self.symplectic_time_step * (pressure -
                       self.detuning_frequency) * self.X, out=self.Y)
                torch.add(self.X, self.symplectic_time_step * (pressure +
                       self.detuning_frequency) * self.Y, out=self.X)
        else:
            a = self.time_step * (pressure - self.detuning_frequency)
            b = self.time_step * (pressure + self.detuning_frequency)
            if pressure < self.detuning_frequency:
                x = torch.sqrt(- a * b)
                cos_coeff = torch.cos(x)
                sinc_coeff = torch.sin(x) / x
                aux_X = cos_coeff * self.X + sinc_coeff * b * self.Y
                aux_Y = cos_coeff * self.Y + sinc_coeff * a * self.X
            else:
                aux_X = self.X + b * self.Y
                aux_Y = self.Y + a * self.X
            self.X = aux_X.copy()
            self.Y = aux_Y.copy()

    @abstractmethod
    def non_symplectic_update(self, ising: Ising) -> None:
        """
        Update the particle vectors with the non-symplectic part of the
        Hamiltonian equations.

        Parameters
        ----------
        ising : Ising
            the Ising model to solve
        """
        raise NotImplementedError

    @final
    def check_stop(self, use_window: bool) -> None:
        """
        Checks the stopping condition and update the `run` attribute
        consequently.

        Parameters
        ----------
        use_window : bool
            indicates whether to use the window as a stopping criterion or not
        verbose : bool, optional
            whether to display evolution information or not (default is True)
        """
        if use_window and self.step % self.sampling_period == 0:
            self.update_window()
            self.run = torch.any(self.stability < self.convergence_threshold - 1)

        if self.step >= self.max_steps:
            self.run = False

    @final
    def get_best_spins(self, ising: Ising, use_window: bool) -> torch.Tensor:
        """
        Retrieves the best spin vector among all the agents.

        Parameters
        ----------
        ising : Ising
            the Ising model to solve
        use_window : bool
            indicates whether to use the window as a stopping criterion or not
        verbose : bool, optional
            whether to display evolution information or not (default is True)

        Returns
        -------
        ground_state : torch.Tensor
            the spin vector giving the lowest Ising energy among all the agents
        """
        if not use_window:
            energies = torch.diag(-.5 * torch.sign(self.X.t()) @ ising.J @
                               torch.sign(self.X) + torch.sign(self.X.t()) @ ising.h)
        else:
            energies = torch.diag(-.5 * self.final_spins.t() @ ising.J @
                               self.final_spins + self.final_spins.t() @ ising.h)

        index = torch.argmin(energies)
        self.agents_progress.close()
        self.iterations_progress.close()

        if not use_window:
            return torch.sign(self.X)[:, index].reshape(-1, 1)
        return self.final_spins[:, index].reshape(-1, 1)

    @final
    def iterate(self, ising: Ising, use_window: bool = True) -> torch.Tensor:
        """
        Iterates the Symplectic Euler Scheme until the stopping condition is
        met.

        Parameters
        ----------
        ising : Ising
            the Ising model to solve
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        verbose : bool, optional
            whether to display evolution information or not (default is True)

        Returns
        -------
        ground_state : torch.Tensor
            the spin vector giving the lowest Ising energy among all the agents
        """
        self.reset(ising)
        start_time = time()

        while self.run:
            self.symplectic_update()
            self.non_symplectic_update(ising)
            self.confine()
            self.step_update()
            self.check_stop(use_window)

        self.time = time() - start_time

        return self.get_best_spins(ising, use_window)


class BallisticHeatedSymplecticEulerScheme(SymplecticEulerScheme):
    """
    Symplectic Euler Scheme for the Heated ballistic Simulated Bifurcation
    (HbSB) algorithm.
    """

    def non_symplectic_update(self, ising: Ising) -> None:
        temp = ising.null_diag_J @ self.X - \
            self.field_coefficient(self.time_step * self.step) * ising.h
        temp = self.xi0 * temp + self.heat_parameter * self.Y
        self.Y += self.time_step * temp


class BallisticSymplecticEulerScheme(SymplecticEulerScheme):
    """
    Symplectic Euler Scheme for the ballistic Simulated Bifurcation (bSB)
    algorithm.
    """

    def non_symplectic_update(self, ising: Ising) -> None:
        temp = ising.null_diag_J @ self.X - \
            self.field_coefficient(self.time_step * self.step) * ising.h
        self.Y += self.time_step * self.xi0 * temp


class DiscreteHeatedSymplecticEulerScheme(SymplecticEulerScheme):
    """
    Symplectic Euler Scheme for the Heated discrete Simulated Bifurcation
    (HdSB) algorithm.
    """

    def non_symplectic_update(self, ising: Ising) -> None:
        temp = ising.null_diag_J @ torch.sign(self.X) - \
            self.field_coefficient(self.time_step * self.step) * ising.h
        temp = self.xi0 * temp + self.heat_parameter * self.Y
        self.Y += self.time_step * temp


class DiscreteSymplecticEulerScheme(SymplecticEulerScheme):
    """
    Symplectic Euler Scheme for the discrete Simulated Bifurcation (dSB)
    algorithm.
    """

    def non_symplectic_update(self, ising: Ising) -> None:
        temp = ising.null_diag_J @ torch.sign(self.X) - \
            self.field_coefficient(self.time_step * self.step) * ising.h
        self.Y += self.time_step * self.xi0 * temp


def main():

    import numpy as np

    if torch.cuda.is_available(): device = torch.device('cuda:0')
    else: device = torch.device('cpu')

    print(f'Environment set on {device}.')

    dim = 1024
    agents = 128
    J = np.random.uniform(-0.5, 0.5, size=(dim, dim))
    h = np.random.uniform(-0.5, 0.5, size=(dim, 1))
    energies = []
    for ballistic in [True, False]:
        for heated in [True, False]:
            ising = Ising(torch.from_numpy(J), torch.from_numpy(h),
            device = device)
            ising.optimize(agents=agents, ballistic=ballistic,
                           heated=heated)
            energies.append(ising.energy)
    print(energies)


if __name__ == "__main__":
    main()
