import numpy as np
from statistics import stdev
from time import time

def symplectic_euler_scheme(
    J,
    h,
    time_step,
    simulation_time,
    kerr_constant,
    detuning_frequency,
    pressure,
    symplectic_parameter
):

    """
    Compute the optimal binary vector with the symplectic Euler's scheme.
    """

    dimension = np.shape(J)[0]

    # Initialization of the oscillators

    X = np.zeros((dimension,1)) 
    Y = np.zeros((dimension,1)) 

    # Introduction of other parameters

    dt = time_step / symplectic_parameter # Symplectic timestep
    number_of_steps = int(simulation_time / time_step)
    xi0 = 0.7 * detuning_frequency / (stdev([J[i][j] for i in range(dimension) for j in range(dimension) if i != j]) * (dimension)**(1/2))

    unit_column = np.ones((dimension, 1))
    diag_J_column = np.array([np.diag(J)]).T

    def A(t):

        p = pressure(t)

        if p < detuning_frequency:

            return 0

        else:

            return ((p - detuning_frequency) / kerr_constant)**(1/2) 

    # Begining of simulation

    start_time = time()

    for step in range(number_of_steps):

        current_time = step * time_step
        current_pressure = pressure(current_time)

        # Symplectic loops

        for _ in range(symplectic_parameter):

            X += dt * (((detuning_frequency + current_pressure) * unit_column - xi0 * diag_J_column) * Y)
            Y -= dt * (X**3 + (detuning_frequency - current_pressure) * X)  

        Y += xi0 * (J @ X - 2 * A(current_time) * h) * time_step

    end_time = time()

    # End of simulation

    simulation_time = end_time - start_time    

    print(f"Run in {simulation_time} seconds.")

    return X