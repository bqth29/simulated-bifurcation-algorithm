from models.Markowitz import Markowitz
from data.data import assets, dates
from time import time
import numpy as np

markowitz = Markowitz.from_csv(assets_list = assets[18:20], number_of_bits = 1, date = dates[-1], risk_coefficient = 10)
markowitz.optimize(
    window_size = 35,
    sample_frequency = 60,
    time_step = 0.01,
    symplectic_parameter = 2,
    pressure = lambda t : 0.0088 * t
)

print(markowitz)
energy = -.5 * markowitz.portfolio['array'].T @ markowitz.covariance @ markowitz.portfolio['array'] + markowitz.portfolio['array'].T @ markowitz.expected_return 
print(energy)
l = []
for i in range(2):
    for j in range(2):
        a = np.array([[i], [j]])
        l.append(-.5 * a.T @ markowitz.covariance @ a + a.T @ markowitz.expected_return)
print(max(l))        

markowitz.pie()
markowitz.table()