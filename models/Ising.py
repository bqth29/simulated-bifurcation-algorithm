import numpy as np
from time import time
from statistics import stdev

class Ising():

    def __init__(self):

        self.J = None
        self.h = None
        self.dimension = None
        self.ground_state = None
        self.matrix = None
        self.ready = False 

    def from_Markowitz(self, Markowitz_model):

        # Basis shift matrix

        matrix = np.zeros((Markowitz_model.number_of_assets * Markowitz_model.number_of_bits, Markowitz_model.number_of_assets))

        for a in range(Markowitz_model.number_of_assets):

            for b in range(Markowitz_model.number_of_bits):

                matrix[a*Markowitz_model.number_of_bits+b][a] = 2**b

        self.matrix = matrix 

        # Spin basis matrix

        sigma_hat = np.block(
            [
                [2**(i+j)*Markowitz_model.covariance for i in range(Markowitz_model.number_of_bits)] for j in range(Markowitz_model.number_of_bits)
            ]
        )

        mu_hat = self.matrix @ Markowitz_model.expected_return

        self.J = -Markowitz_model.risk_coefficient/2 * sigma_hat
        self.h = Markowitz_model.risk_coefficient/2 * sigma_hat @ np.ones((Markowitz_model.number_of_assets * Markowitz_model.number_of_bits, 1)) - mu_hat
        self.dimension = Markowitz_model.number_of_assets * Markowitz_model.number_of_bits

    def assess_ready(self):

        """
        Checks if the Ising model can be computed.
        """

        if self.J is not None and self.h is not None:

            try:

                J_x, J_y = np.shape(self.J)
                h_x, h_y = np.shape(self.h)

                self.ready = (J_x == J_y) and (J_x == h_x) and (h_y == 1)

            except:

                self.ready = False    

        else:

            self.ready = False    
    
    def optimize(self, Hamiltonian, parameters):

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  

        self.assess_ready()  

        if self.ready:

            # Initialization of the oscillators

            X = np.zeros((self.dimension,1)) 
            Y = np.zeros((self.dimension,1)) 

            # Introduction of other parameters

            dt = parameters.time_step / parameters.symplectic_parameter # Symplectic timestep
            number_of_steps = int(parameters.simulation_time / parameters.time_step)
            xi0 = 0.7 * Hamiltonian.detuning_frequency / (stdev([self.J[i][j] for i in range(self.dimension) for j in range(self.dimension) if i != j]) * (self.dimension)**(1/2))

            unit_column = np.ones((self.dimension, 1))
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

                current_time = step * parameters.time_step
                current_pressure = Hamiltonian.pressure(current_time)

                # Symplectic loops

                for _ in range(parameters.symplectic_parameter):

                    X += dt * (((Hamiltonian.detuning_frequency + current_pressure) * unit_column - xi0 * diag_J_column) * Y)
                    Y -= dt * (X**3 + (Hamiltonian.detuning_frequency - current_pressure) * X)  

                Y += xi0 * (self.J @ X - 2 * A(current_time) * self.h) * parameters.time_step

            end_time = time()

            # End of simulation

            parameters.simulation_time = end_time - start_time    

            print(f"Run in {parameters.simulation_time} seconds.")

            self.ground_state = X

        else:

            raise Exception("The Ising model is not well defined. Please verify its definition.")
