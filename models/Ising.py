import numpy as np
from models.SymplecticEulerScheme import SymplecticEulerScheme

class Ising():

    """
    Implementation of Ising problems.
    """

    def __init__(self, J : np.ndarray, h : np.ndarray) -> None:

        """
        Class constructor.
        """

        self.J = J
        self.h = h
        self.ground_state = None
        self.energy = None      
    
    def optimize(
        self,
        kerr_constant : float = 1,
        detuning_frequency : float = 1,
        pressure = lambda t : 0.01 * t,
        time_step : float = 0.01,
        symplectic_parameter : int = 2,
        simulation_time : int = 600,
        window_size = 50,
        stop_criterion = True,
        check_frequency : int = 1000,
    ) -> None:

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  

        if self.ground_state is None:

            euler = SymplecticEulerScheme(
                self.J,
                self.h,
                kerr_constant = kerr_constant,
                detuning_frequency = detuning_frequency,
                pressure = pressure,
                time_step = time_step,
                symplectic_parameter = symplectic_parameter,
                simulation_time = simulation_time,
                window_size = window_size,
                stop_criterion = stop_criterion,
                check_frequency = check_frequency
            )
            
            self.ground_state = np.sign(euler.run())
            energy = -0.5 * self.ground_state.T @ self.J @ self.ground_state + self.ground_state.T @ self.h
            self.energy = energy[0][0]