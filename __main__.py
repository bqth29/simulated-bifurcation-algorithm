from models.Markowitz import Markowitz
from data.data import assets, dates
from time import time
import numpy as np

markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 4, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(
    convergence_threshold = 35,
    sampling_period = 60,
    time_step = 0.01,
    symplectic_parameter = 2,
    pressure = lambda t : 0.0088 * t
)

print(markowitz)       

# markowitz.pie()
# markowitz.table()