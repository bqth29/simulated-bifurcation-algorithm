from typing import Tuple, final
import numpy as np
from time import time

# Classes

class Ising():

    def __init__(self, J: np.ndarray, h: np.ndarray, assert_parameters: bool = True) -> None:

        """
        Class constructor.
        """
        
        self.J                  = J
        self.h                  = h

        self.dimension          = J.shape[0]
        
        self.ground_state       = None 

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

    def energy(self) -> float:

        if self.ground_state is None:
            return None

        else:
            energy = -0.5 * self.ground_state.T @ self.J @ self.ground_state + self.ground_state.T @ self.h
            return energy[0][0]     

    def get_ground_state(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
        display_time: bool = True,
    ) -> None:

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  
        
        start_time = time()

        X, _ = symplectic_euler_scheme(
            self,
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period
        )

        end_time = time()

        simulation_time = round(end_time - start_time, 3)

        if display_time:    

            print(f"Run in {simulation_time} seconds.")

        self.ground_state = np.sign(X)

class SBModel():

    def __to_Ising__(self) -> Ising:
        pass

    def __from_Ising__(self, ising: Ising) -> None:
        pass
    
    @final
    def optimize(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
        display_time: bool = True,
    ) -> None:

        ising_equivalent = self.__to_Ising__()
        ising_equivalent.get_ground_state(
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period,
            display_time,
        )
        self.__from_Ising__(ising_equivalent)     

# Euler Scheme

def symplectic_euler_scheme(
    ising: Ising,
    detuning_frequency: float = 1,
    kerr_constant: float = 1,
    pressure = lambda t: 0.01 * t,
    time_step: float = 0.01,
    symplectic_parameter: int = 2,
    convergence_threshold: int = 35,
    sampling_period: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Symplectic Euler scheme computing the evolution of the oscillators of a KNPO network.
    Ends using a rolling-window stop criterion.
    """

    # Parameters initialization

    X = np.zeros((ising.dimension, 1)) 
    Y = np.zeros((ising.dimension, 1)) 

    xi0 = 0.7 * detuning_frequency / (np.std(ising.J) * (ising.dimension)**(1/2))

    current_spins = np.zeros((ising.dimension, 1))
    equal_streak  = 0

    symplectic_time_step = time_step / symplectic_parameter

    step = 0

    # Simulation

    while equal_streak < convergence_threshold - 1:

        factor = pressure(step * time_step) - detuning_frequency

        if factor > 0:

            # Symplectic loops

            for _ in range(symplectic_parameter):

                X += symplectic_time_step * detuning_frequency * Y
                Y -= symplectic_time_step * (kerr_constant * X**3 - factor * X)  

            Y += time_step * xi0 * (ising.J @ X - 2 * pow(factor / kerr_constant, .5) * ising.h)

            # Check the stop criterion

            if step % sampling_period == 0:

                spins = np.sign(X)
                
                if np.allclose(spins, current_spins):
                    equal_streak += 1
                else:
                    equal_streak = 0
                    current_spins = spins.copy()

        step += 1      

    return X, Y