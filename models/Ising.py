import numpy as np
from time import time
from statistics import stdev
from models.Hamiltionian import Hamiltonian

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
    
    def optimize(self, Hamiltonian : Hamiltonian = Hamiltonian()) -> None:

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  

        if self.ground_state is None:

            dimension = np.shape(self.J)[0]

            # Initialization of the oscillators

            X = np.zeros((dimension,1)) 
            Y = np.zeros((dimension,1)) 

            # Introduction of other parameters

            dt = Hamiltonian.time_step / Hamiltonian.symplectic_parameter # Symplectic timestep
            number_of_steps = int(Hamiltonian.simulation_time / Hamiltonian.time_step)
            xi0 = 0.7 * Hamiltonian.detuning_frequency / (stdev([self.J[i][j] for i in range(dimension) for j in range(dimension) if i != j]) * (dimension)**(1/2))

            unit_column = np.ones((dimension, 1))
            diag_J_column = np.array([np.diag(self.J)]).T

            def A(t):

                p = Hamiltonian.pressure(t)

                if p < Hamiltonian.detuning_frequency:

                    return 0

                else:

                    return ((p - Hamiltonian.detuning_frequency) / Hamiltonian.kerr_constant)**(1/2) 

            # Begining of simulation

            start_time = time()

            for step in range(number_of_steps):

                current_time = step * Hamiltonian.time_step
                current_pressure = Hamiltonian.pressure(current_time)

                # Symplectic loops

                for _ in range(Hamiltonian.symplectic_parameter):

                    X += dt * (((Hamiltonian.detuning_frequency + current_pressure) * unit_column - xi0 * diag_J_column) * Y)
                    Y -= dt * (X**3 + (Hamiltonian.detuning_frequency - current_pressure) * X)  

                Y += xi0 * (self.J @ X - 2 * A(current_time) * self.h) * Hamiltonian.time_step

            end_time = time()

            # End of simulation

            simulation_time = end_time - start_time    

            print(f"Run in {simulation_time} seconds.")

            self.ground_state = np.sign(X)