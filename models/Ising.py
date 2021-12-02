import numpy as np
from time import time

class Ising():

    """
    Implementation of Ising problems.
    """

    def __init__(self, J : np.ndarray, h : np.ndarray) -> None:

        """
        Class constructor.
        """

        # Checking types
        assert isinstance(J, np.ndarray), f"The type of J must be a numpy array, instead got {type(J)}"
        assert isinstance(h, np.ndarray), f"The type of h must be a numpy array, instead got {type(h)}"

        # Checking dimensions
        assert J.shape[0] == J.shape[1], f"J must be a square matrix, instead got {J.shape}"
        assert h.shape[0] == J.shape[0], f"The dimension of h must fits J's, instead of {J.shape[0]} got {h.shape[0]}"
        assert h.shape[1] == 1, f"h must be a column vector with dimensions of this pattern: (n,1), instead got {h.shape}"

        # Checking J's properties
        assert np.allclose(J, J.T), "J must be symmetric"
        assert not np.any(J == np.zeros(J.shape)), "J must not have null elements"

        self.J = J
        self.h = h
        self.dimension = J.shape[0]
        self.ground_state = None
        self.energy = None   

    def optimize(
        self,
        kerr_constant : float = 1,
        detuning_frequency : float = 1,
        pressure = lambda t : 0.0088 * t,
        time_step : float = 0.01,
        symplectic_parameter : int = 2,
        window_size = 35,
        sampling_period : int = 60,
        display_time : bool = True,
    ) -> None:

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  

        if self.ground_state is None:

            # Symplectic parameter

            symplectic_time_step = time_step / symplectic_parameter

            # Parameters calculated from matrix
            
            xi0 = 0.7 * detuning_frequency / (np.std(self.J) * (self.dimension)**(1/2))

            # Initialization of the oscillators

            X = np.zeros((self.dimension,1)) 
            Y = np.zeros((self.dimension,1)) 

            # Definition of the window

            window = np.zeros((self.dimension, window_size), dtype=np.float64)
            zeros = np.zeros((self.dimension,))
            zeros_matrix = np.zeros((self.dimension, window_size), dtype=np.float64)

            # Begining of simulation

            run = True
            step = 0

            start_time = time()

            while run:

                factor = pressure(step * time_step) - detuning_frequency

                if True:

                    for _ in range(symplectic_parameter):

                        X += symplectic_time_step * detuning_frequency * Y
                        Y -= symplectic_time_step * (kerr_constant * X**3 - factor * X)  

                    Y += time_step * xi0 * (self.J @ X - 2 * pow(max(factor, 0) / kerr_constant, .5) * self.h)

                    # Check the stop criterion

                    if step % sampling_period == 0:

                        window = np.roll(window, -1, axis = 1)
                        window[:, -1] = np.sign(X).T[0]
                        variance = np.var(window, axis = 1)

                        run = not np.allclose(variance, zeros, rtol = 1e-8, atol = 1e-8) or np.any(np.isclose(window, zeros_matrix))

                step += 1         

            end_time = time()

            # End of simulation

            simulation_time = round(end_time - start_time, 3)

            if display_time:    

                print(f"Run in {round(simulation_time,3)} seconds.")

            self.ground_state = np.sign(X)    
            energy = -0.5 * self.ground_state.T @ self.J @ self.ground_state + self.ground_state.T @ self.h
            self.energy = energy[0][0]   