import numpy as np

def vector_to_binary_basis_matrix(number_of_assets, number_of_bits):

    """
    Return the matrix one needs to multiple a vector by to write it
    in the binary basis (for a given number of bits). 
    """

    matrix = np.zeros((number_of_assets * number_of_bits, number_of_assets))

    for a in range(number_of_assets):
        for b in range(number_of_bits):

            matrix[a*number_of_bits+b][a] = 2**b

    return matrix       

def to_ising(sigma, mu, number_of_bits, gamma = 1):

    """
    Takes in entry a SPD matrix sigma and a weight vector mu and transform
    them in the spin basis (for a given number of bits) to turn the problem
    into an Ising one.
    """

    number_of_assets = np.shape(sigma)[0]
    new_basis_matrix = vector_to_binary_basis_matrix(number_of_assets, number_of_bits)

    sigma_hat = np.block(
        [
            [2**(i+j)*sigma for i in range(number_of_bits)] for j in range(number_of_bits)
        ]
    )
    mu_hat = new_basis_matrix @ mu

    J = -gamma/2 * sigma_hat
    h = gamma/2 * sigma_hat @ np.ones((number_of_assets * number_of_bits, 1)) - mu_hat

    return J, h 