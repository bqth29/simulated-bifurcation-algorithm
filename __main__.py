from models.Markowitz import Markowitz
from models.Markowitz import recursive_subportfolio_optimization as RSO
from data.data import assets, dates
from time import time
import numpy as np
import random as rd
from MARSO import MultiAgentRecursiveSubportfolioOptimizer as MARSO

l = assets#rd.sample(assets, 440)
bits = 4

marso = MARSO([4, 2, 1], l, number_of_bits=bits)
marso.solve()
print(marso.utility_function)
print(marso.portfolio)

print(RSO(l, bits))

# markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 2, date = dates[-1], risk_coefficient = 1)
# markowitz.optimize(
#     convergence_threshold = 35,
#     sampling_period = 60,
#     time_step = 0.01,
#     symplectic_parameter = 2,
#     pressure = lambda t : 0.0088 * t
# )

# print(markowitz)

# markowitz.pie()
# markowitz.table()