import numpy as np

from ising import to_ising, vector_to_binary_basis_matrix

def all_binary_vectors(dimension):

    vectors_list = []

    for i in range(2**dimension):

        # txt_binary = bin(i)[2:]
        # list_binary = [float(txt_binary[x]) for x in range(len(txt_binary))] + [0. for _ in range(dimension - len(txt_binary))]
        # list_binary.reverse()
        # vectors_list.append(np.array([list_binary]).T)

        txt_binary = bin(i)[2:]
        list_binary = [0 for _ in range(dimension - len(txt_binary))] + list(txt_binary)
        aux_list = [int(b) for b in list_binary]
        vectors_list.append(np.array(aux_list).reshape((dimension,1)))

    return vectors_list

def brute_force(J,h):

    dimension = np.shape(J)[0]
    vectors_list = all_binary_vectors(dimension)

    min_value = float('inf')
    best_ground_state = None

    for vector in vectors_list:

        spin_vector = 2 * vector - 1
        
        ising_energy = -0.5 * spin_vector.T @ J @ spin_vector + h.T @ spin_vector
        if ising_energy[0][0] < min_value:

            min_value = ising_energy[0][0]
            best_ground_state = spin_vector.copy()
    #print(f'Ising BF energy: {min_value} ({best_ground_state})')
    return best_ground_state, min_value

def optimal_portfolio(sigma, mu, number_of_bits, gamma = 1):

    number_of_assets = np.shape(sigma)[0]

    # Shifting to the spin frame

    J,h = to_ising(sigma, mu, number_of_bits, gamma)   

    # Retreiving the binary vector   
    # 
    bf, energy = brute_force(J,h)

    best_binary = (1. + bf) / 2.
    #print(f'BF bin: {best_binary}')
    portfolio = np.transpose(vector_to_binary_basis_matrix(number_of_assets, number_of_bits)) @ best_binary

    return portfolio.T[0], energy

# sigma = np.eye(2)
# mu = np.random.random((2,1))
# number_of_bits = 5    

# print(optimal_portfolio(sigma, mu, number_of_bits))
