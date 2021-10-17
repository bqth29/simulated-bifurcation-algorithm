import numpy as np
from models.SymplecticEulerSchemeStop import SymplecticEulerScheme

class Ising():

    """
    Implementation of Ising problems.
    """

    def __init__(self, J : np.ndarray, h : np.ndarray) -> None:

        """
        Class constructor. Checks if the arguments provided match the requirement of an Ising problem.
        """

        # J and h must be numpy arrays

        if not isinstance(J, np.ndarray):

            raise TypeError(f"J must be a numpy array. Instead its class is {type(J)}.")

        elif not isinstance(h, np.ndarray):

            raise TypeError(f"h must be a numpy array. Instead its class is {type(h)}.")  

        # J must be definite symetric positive    

        elif not np.all(abs(J - J.T) < 10**-12):

            raise ValueError("J must be symetric.")  

        # elif min(np.linalg.eigvals(J)) <= 0 and max(np.linalg.eigvals(J)) >= 0:

        #     raise ValueError("J must be positive definite.")

        # h must be a column vector    

        elif np.shape(h)[1] != 1:

            raise ValueError(f"h must be a column vector, i.e. its dimensions must fit the following pattern: (n,1). Instead, its dimensions are {np.shape(h)}.")

        # J and h dimensions must fit
        
        elif np.shape(J)[0] != np.shape(h)[0]:

            raise ValueError(f"J and h dimensions must fit. However, J is a square matrix of size {np.shape(J)[0]} and h is a column vector of size {np.shape(h)[0]}.")   

        else:

            self.J = J
            self.h = h
            self.ground_state = None       
    
    def optimize(
        self,
        kerr_constant : float = 1,
        detuning_frequency : float = 1,
        pressure = lambda t : 0.01 * t,
        time_step : float = 0.01,
        symplectic_parameter : int = 2
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
                symplectic_parameter = symplectic_parameter
            )
            self.ground_state = euler.run()