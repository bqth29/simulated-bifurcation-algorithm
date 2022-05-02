from models.Markowitz import Markowitz
from models.Markowitz import recursive_subportfolio_optimization as RSO
from data.data import assets, dates
from time import time
import numpy as np
import random as rd

import simulated_bifurcation as sb

#print(RSO(assets, 1))

sb.Ising.set_env(
    detuning_frequency=2
)

markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(
    convergence_threshold = 35,
    sampling_period = 60,
    time_step = 0.01,
    symplectic_parameter = 2
)

print(markowitz)
print(markowitz.optimization_logs)

# markowitz.pie()
# markowitz.table()