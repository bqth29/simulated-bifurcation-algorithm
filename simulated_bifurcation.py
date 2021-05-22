import numpy as np

from euler import symplectic_euler_scheme
from ising import to_ising, vector_to_binary_basis_matrix

def simulated_bifurcation(
    sigma, 
    mu, 
    number_of_bits,
    time_step = 0.01,
    simulation_time = 600,
    kerr_constant = 1,
    detuning_frequency = 1,
    risk_coefficient = 1,
    pressure = lambda t: 0.01 * t,
    symplectic_parameter = 2
):

    """
    Computes the optimal portfolio.
    """

    number_of_assets = np.shape(sigma)[0]

    # Shifting to the spin frame

    J,h = to_ising(sigma, mu, number_of_bits, gamma = risk_coefficient)

    # Euler scheme

    X = symplectic_euler_scheme(J, h, time_step, simulation_time, kerr_constant, 
                                detuning_frequency, pressure, symplectic_parameter)

    portfolio = np.transpose(vector_to_binary_basis_matrix(number_of_assets, number_of_bits)) @ (1 + np.sign(X))/2 

    return portfolio.T[0]



