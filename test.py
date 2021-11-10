from models.Markowitz import Markowitz
from data.data import assets, dates
from time import time
import numpy as np

from brute_force import brute_force, optimal_portfolio, all_binary_vectors
from random import sample

import json

PERFORMANCES_14_ASSETS_1_BIT = {}

for period in range(20,50):
    for window in range(45,90):

        diff = 0
        times = []
        accuracies = []

        for _ in range(1):

            markowitz = Markowitz.from_csv(assets_list = sample(assets,14), number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
            markowitz.optimize(
                window_size = window,
                sampling_period = period,
                time_step = 0.01,
                symplectic_parameter = 2,
                pressure = lambda t : 0.0088 * t,
                display_time = False
            )
            a = markowitz.gain()
            sigma, mu = markowitz.covariance, markowitz.expected_return

            bf, bf_energy = optimal_portfolio(sigma, mu, markowitz.number_of_bits,  markowitz.risk_coefficient)
            portfolio = np.array(bf).reshape((markowitz.number_of_assets,1))
            b = -0.5 * markowitz.risk_coefficient * portfolio.T @ sigma @ portfolio + mu.T @ portfolio

            acc = np.sum(np.equal(markowitz.portfolio['array'], portfolio))

            accuracies.append(int(acc))
            times.append(float(markowitz.run_in))
            
        PERFORMANCES_14_ASSETS_1_BIT[str(period) + ' / ' + str(window)] = {
            'accuracy': accuracies[:],
            'time': times[:]
        }

with open('perf_14a_1b.json', 'w') as f:
    json.dump(PERFORMANCES_14_ASSETS_1_BIT, f)   