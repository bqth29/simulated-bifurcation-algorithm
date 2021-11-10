from models.Markowitz import Markowitz
from data.data import assets, dates
from time import time
import numpy as np

from brute_force import brute_force, optimal_portfolio, all_binary_vectors
from random import sample

markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(
    window_size = 40,
    sampling_period = 50,
    time_step = 0.01,
    symplectic_parameter = 2,
    pressure = lambda t : 0.0088 * t
)
# diff = 0
# for _ in range(100):
#     markowitz = Markowitz.from_csv(assets_list = sample(assets,14), number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
#     markowitz.optimize(
#         window_size = 50,
#         sampling_period = 50,
#         time_step = 0.01,
#         symplectic_parameter = 2,
#         pressure = lambda t : 0.0088 * t
#     )
#     a = markowitz.gain()
#     sigma, mu = markowitz.covariance, markowitz.expected_return


#     bf, bf_energy = optimal_portfolio(sigma, mu, markowitz.number_of_bits,  markowitz.risk_coefficient)
#     portfolio = np.array(bf).reshape((markowitz.number_of_assets,1))
#     b = -0.5 * markowitz.risk_coefficient * portfolio.T @ sigma @ portfolio + mu.T @ portfolio
    
#     if a < b:
#         diff += 1
#         c = abs(a-b)/b
#         print(f'SB: {a} / BF: {b[0][0]} ({round(c[0][0],3)} rel.)')

#     elif a > b:

#         print('Erreur')   

# print(f'Hard accuracy: {100-diff}%')     

print(markowitz)
print(f'Gain: {markowitz.gain()}')   

# markowitz.pie()
# markowitz.table()