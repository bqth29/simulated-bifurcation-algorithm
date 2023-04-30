from abc import ABC, abstractmethod
from typing import List, Tuple, Union, final

import torch
from tqdm import tqdm
from numpy import minimum, argmin


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

        spins = optimizer.symplectic_update(self, use_window)
        
        if self.linear_term: self.computed_spins = spins[-1] * spins[:-1, :]
        else: self.computed_spins = spins


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
            convergence_threshold,
            sampling_period,
            max_steps,
            agents,
            pressure_slope,
            gerschgorin,
            use_window,
            ballistic,
            heat_parameter,
            verbose
        )
        self.__from_Ising__(ising_equivalent)


class Optimizer():
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

    ballistic_activation = torch.nn.Identity()
    discrete_activation = torch.sign

    def __init__(
        self,
        time_step: float,
        convergence_threshold: int,
        sampling_period: int,
        max_steps: int,
        agents: int,
        pressure_slope: float,
        gerschgorin: bool,
        ballistic: bool,
        heat_parameter: float,
        verbose: bool
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

        # Optimizer setting
        self.ballistic = ballistic
        self.discrete = not ballistic

        if ballistic: self.activation = Optimizer.ballistic_activation
        else: self.activation = Optimizer.discrete_activation

        self.heat_parameter = heat_parameter
        self.heated = heat_parameter is not None

        # Simulation parameters
        self.initialized = False
        self.time_step = time_step
        self.agents = agents

        # Stopping criterion parameters
        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps

        # Quantum parameters
        self.gerschgorin = gerschgorin
        self.pressure = lambda t: minimum(
                pressure_slope * t, 1.)

        # Evolutive parameters
        self.X, self.Y = None, None
        self.dimension = None
        self.ising_model = None
        self.current_spins = None
        self.final_spins = None
        self.stability = None
        self.bifurcated = None
        self.previously_bifurcated = None
        self.new_bifurcated = None
        self.equal = None
        self.run = True
        self.xi0 = None
        self.step = 0
        self.time = 0

    @property
    def shape(self) -> Tuple[Union[int, None], int]:
        if self.initialized: return (self.dimension, self.agents)
        else: return (None, self.agents)

    def set_ballistic(self) -> None:
        if self.ballistic: pass
        else:
            self.discrete = False
            self.ballistic = True
            self.activation = Optimizer.ballistic_activation
    
    def set_discrete(self) -> None:
        if self.discrete: pass
        else:
            self.discrete = True
            self.ballistic = False
            self.activation = Optimizer.discrete_activation

    def confine(self) -> None:
        """
        Confine the particles' position in the range [-1, 1], i.e. if a `x > 1`
        or `x < -1`, `x` is replaced by `sign(x)` and the corresponding
        pulsation `y` is set to 0.
        """
        self.Y[torch.abs(self.X) > 1.] = 0
        torch.clip(self.X, -1., 1., out=self.X)

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

    def reset(self, ising: Ising) -> None:
        """
        Reset the simulation parameters.

        Parameters
        ----------
        ising : Ising
            the Ising model to solve
        """
        self.dimension = ising.matrix.shape[0]
        self.ising_model = Ising.remove_diagonal(ising.matrix)

        self.X = 2 * torch.rand(size=(self.dimension, self.agents), 
                device = ising.device, dtype=ising.dtype) - 1
        self.Y = 2 * torch.rand(size=(self.dimension, self.agents),
                device = ising.device, dtype=ising.dtype) - 1

        # Stopping window

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

        if not self.gerschgorin:
            self.xi0 = 0.7 / \
                (torch.std(self.ising_model) * (self.dimension)**(1/2))
        else:
            self.xi0 = 1. / torch.max(
                torch.sum(torch.abs(self.ising_model), axis=1))

        self.initialized = True

    def step_update(self) -> None:
        """
        Increments the current step by 1.
        """
        self.step += 1
        self.iterations_progress.update()

    def update_X(self) -> None:
        pressure = self.pressure(self.time_step * self.step)
        torch.add(self.X, self.time_step * (1. + pressure) * self.Y, out=self.X)
    
    def update_Y(self) -> None:
        pressure = self.pressure(self.time_step * self.step)
        torch.add(
            self.Y,
            self.time_step * (pressure - 1.) * self.X,
            out=self.Y
        )

    def quadratic_update(self) -> None:
        temp = self.ising_model @ self.activation(self.X)
        torch.add(
            self.Y,
            self.time_step * self.xi0 * temp,
            out=self.Y
        )

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

    def symplectic_update(self, ising: Ising, use_window: bool) -> torch.Tensor:
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

        while self.run:

            if self.heated: heatY = self.Y.clone().detach()

            self.update_Y()
            self.update_X()
            self.quadratic_update()
            self.confine()

            if self.heated: torch.add(self.Y, self.time_step * self.heat_parameter * heatY, out=self.Y)

            self.step_update()
            self.check_stop(use_window)

        self.agents_progress.close()
        self.iterations_progress.close()

        if use_window: 
            if torch.any(self.bifurcated):
                return self.final_spins[:, self.bifurcated]
            else:
                print('No agent bifurcated. Returned final oscillators instead.')
                return torch.sign(self.X)
        else: return torch.sign(self.X)


def main():

    import numpy as np

    if torch.cuda.is_available(): device = torch.device('cuda:0')
    else: device = torch.device('cpu')

    print(f'Environment set on {device}.')

    dim = 400
    agents = 128
    J = np.random.uniform(-0.5, 0.5, size=(dim, dim))
    J = .5 * (J + J.T)
    h = np.random.uniform(-0.5, 0.5, size=(dim, 1))
    energies = []
    for ballistic in [True, False]:
        for gerschgorin in [True, False]:
            ising = Ising(torch.from_numpy(J), torch.from_numpy(h),
            device = device)
            ising.optimize(agents=agents, ballistic=ballistic,
                gerschgorin=gerschgorin)
            energies.append(ising.energy)
    print(energies)


if __name__ == "__main__":
    main()
