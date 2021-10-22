import numpy as np
from time import time

class SymplecticEulerScheme():

    def __init__(
        self, 
        matrix : np.ndarray, 
        vector : np.ndarray,
        kerr_constant : float = 1,
        detuning_frequency : float = 1,
        pressure = lambda t : 0.01 * t,
        time_step : float = 0.01,
        symplectic_parameter : int = 2,
        window_size = 50,
        simulation_time : int = 600,
        stop_criterion : bool = True,
        check_frequency : int = 1000,
    ) -> None:
    
        # Data
        
        self.matrix = matrix
        self.vector = vector 
        self.dimension = np.shape(matrix)[0]
 
        # Hamiltonian parameters
 
        self.kerr_constant = kerr_constant
        self.detuning_frequency = detuning_frequency

        # Hamiltonian functions

        self.pressure = pressure
        self.A = lambda t : 0 if (self.pressure(t) < self.detuning_frequency) else ((self.pressure(t) - self.detuning_frequency) / self.kerr_constant)**(1/2)
        
        # Simulation parameters
        
        self.time_step = time_step
        self.window_size = window_size
        self.simulation_time = simulation_time
        self.number_of_steps = simulation_time // time_step
        self.stop_criterion = stop_criterion
        self.check_frequency = check_frequency

        # Symplectic parameter

        self.symplectic_parameter = symplectic_parameter
        self.symplectic_time_step = time_step / symplectic_parameter

        # Parameters calculated from matrix
        
        self.xi0 = 0.7 * detuning_frequency / (np.std(self.matrix - np.diag(np.diag(self.matrix))) * (self.dimension)**(1/2))

    def first_positive(self):
        t = 0
        while(self.pressure(t * self.time_step) < self.detuning_frequency):
            t += 1   
        return t      

    def run(
        self, 
        display_time = True
    ):

        # Initialization of the oscillators

        X = np.zeros((self.dimension,1)) 
        Y = np.zeros((self.dimension,1)) 

        # Definition of the window

        if self.stop_criterion:

            window = np.zeros((self.dimension, self.window_size), dtype=np.float64)
            zeros = np.zeros((self.dimension,))
            zeros_matrix = np.zeros((self.dimension, self.window_size), dtype=np.float64)

        # Begining of simulation

        criterion = True
        step = self.first_positive()

        start_time = time()

        while criterion:

            current_time = self.time_step * step
            current_pressure = self.pressure(current_time)

            # Symplectic loops

            for _ in range(self.symplectic_parameter):

                X += self.symplectic_time_step * self.detuning_frequency * Y
                Y -= self.symplectic_time_step * (X**3 + (self.detuning_frequency - current_pressure) * X)  

            Y += self.xi0 * (self.matrix @ X - 2 * self.A(current_time) * self.vector) * self.time_step

            # Check the stop criterion
            
            step += 1   

            if self.stop_criterion:

                window = np.roll(window, -1, axis = 1)
                window[:, -1] = np.sign(X).T[0]
                variance = np.var(window, axis = 1)

                if step % self.check_frequency == 0:
                    criterion = not np.allclose(variance, zeros, rtol = 1e-8, atol = 1e-8)# or np.any(np.equal(window, zeros_matrix)) 

            else:

                criterion = step < self.number_of_steps    

        end_time = time()

        # End of simulation

        simulation_time = round(end_time - start_time, 3)

        if display_time:    

            print(f"Run in {round(simulation_time,3)} seconds.")

        # Returning the result    

        return X   