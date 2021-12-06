from typing import Tuple
import numpy as np
from time import time

class KNPO():

    """
    Implementation of Ising problems through a Kerr-nonlinear parametric oscillator network.
    """

    def __init__(
        self,
        J: np.ndarray,
        h: np.ndarray,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        assert_parameters: bool = True,
    ) -> None:

        """
        Class constructor.
        """
        
        # Magnetic interactions

        self.J                  = J
        self.h                  = h

        self.dimension          = J.shape[0]

        # Hamiltonian parameters

        self.detuning_frequency = detuning_frequency
        self.kerr_constant      = kerr_constant
        self.pressure           = pressure

        self.xi0                = 0.7 * detuning_frequency / (np.std(self.J) * (self.dimension)**(1/2))

        # Ising-related values
        
        self.ground_state       = None
        self.energy             = None  

        # Check parameters

        if assert_parameters:
            self.__assert__()

    def __assert__(self) -> None:

        """
        Asserts that the parameters of the object follow the right pattern.
        """  

        # Checking types
        assert isinstance(self.J, np.ndarray), f"WARNING: The type of J must be a numpy array, instead got {type(self.J)}"
        assert isinstance(self.h, np.ndarray), f"WARNING: The type of h must be a numpy array, instead got {type(self.h)}"

        # Checking dimensions
        assert self.J.shape[0] == self.J.shape[1], f"WARNING: J must be a square matrix, instead got {self.J.shape}"
        assert self.h.shape[0] == self.J.shape[0], f"WARNING: The dimension of h must fits J's, instead of {self.J.shape[0]} got {self.h.shape[0]}"
        assert self.h.shape[1] == 1, f"WARNING: h must be a column vector with dimensions of this pattern: (n,1), instead got {self.h.shape}"

        # Checking J's properties
        assert np.allclose(self.J, self.J.T), "WARNING: J must be symmetric"
        assert not np.any(self.J == np.zeros(self.J.shape)), "WARNING: J must not have null elements"

    def __stop_criterion__(self, spins) -> bool:

        """
        Determines whether the Euler's scheme must stop.
        """     
        
        self.window = np.roll(self.window, -1, axis = 1)
        self.window[:, -1] = spins
        variance = np.var(self.window, axis = 1)

        return not np.allclose(variance, np.zeros((self.dimension,)), rtol = 1e-8) or np.any(self.window == 0)

    def __symplectic_euler_scheme__(
        self,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        window_size: int = 35,
        sampling_period: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Symplectic Euler scheme computing the evolution of the oscillators of the network.
        Ends using the stop criterion defined above.
        """

        # Parameters initialization

        X = np.zeros((self.dimension,1)) 
        Y = np.zeros((self.dimension,1)) 

        self.window = np.zeros((self.dimension, window_size), dtype=np.float64)
        symplectic_time_step = time_step / symplectic_parameter

        run = True
        step = 0

        # Simulation

        while run:

            factor = self.pressure(step * time_step) - self.detuning_frequency

            if factor > 0:

                # Symplectic loops

                for _ in range(symplectic_parameter):

                    X += symplectic_time_step * self.detuning_frequency * Y
                    Y -= symplectic_time_step * (self.kerr_constant * X**3 - factor * X)  

                Y += time_step * self.xi0 * (self.J @ X - 2 * pow(factor / self.kerr_constant, .5) * self.h)

                # Check the stop criterion

                if step % sampling_period == 0:

                    run = self.__stop_criterion__(np.sign(X).T[0])

            step += 1      

        return X, Y    

    def optimize(
        self,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        window_size: int = 35,
        sampling_period: int = 50,
        display_time : bool = True,
    ) -> None:

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  
        
        # Beginning of the simulation

        start_time = time()

        X, _ = self.__symplectic_euler_scheme__(
            time_step, symplectic_parameter, window_size, sampling_period
        )

        end_time = time()

        # End of the simulation

        simulation_time = round(end_time - start_time, 3)

        if display_time:    

            print(f"Run in {simulation_time} seconds.")

        self.ground_state = np.sign(X)    
        energy = -0.5 * self.ground_state.T @ self.J @ self.ground_state + self.ground_state.T @ self.h
        self.energy = energy[0][0]   