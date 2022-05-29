from models.Markowitz import Markowitz
from models.Markowitz import recursive_subportfolio_optimization as RSO
from data.data import assets, dates
from time import time
import numpy as np
import random as rd

print(RSO(rd.sample(assets, 227), 3))

markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(
    convergence_threshold = 35,
    sampling_period = 60,
    time_step = 0.01,
    symplectic_parameter = 2
)

print(markowitz)

# markowitz.pie()
# markowitz.table()