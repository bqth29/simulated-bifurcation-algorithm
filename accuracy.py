from models.Markowitz import Markowitz
from data.data import assets, dates
from time import time
import numpy as np

from brute_force import brute_force, optimal_portfolio, all_binary_vectors
from random import sample

import json

DATA  = {}
LOOPS = 100

def relat(exact, approx):
    if exact == approx:
        return 0.
    else:
        return abs((exact - approx) / exact)

for n_assets in range(2,16):

    for n_bits in range(1,8):

        if n_assets * n_bits < 15:

            ising_acc     = []
            markowitz_acc = []
            exact         = 0

            for _ in range(LOOPS):

                markowitz = Markowitz.from_csv(
                    assets_list = sample(assets, n_assets),
                    number_of_bits = n_bits,
                )
                markowitz.optimize(
                    display_time = False
                )

                approx_ising     = markowitz.ising_energy
                approx_markowitz = markowitz.gain()

                sigma, mu = markowitz.covariance, markowitz.expected_return

                bf, exact_ising = optimal_portfolio(sigma, mu, markowitz.number_of_bits,  markowitz.risk_coefficient)
                portfolio = np.array(bf).reshape((markowitz.number_of_assets,1))
                exact_markowitz = -0.5 * markowitz.risk_coefficient * portfolio.T @ sigma @ portfolio + mu.T @ portfolio

                if isinstance(exact_markowitz, np.ndarray):
                    exact_markowitz = exact_markowitz[0][0]

                exact += int(approx_ising == exact_ising)   
                ising_acc.append(relat(exact_ising, approx_ising))
                markowitz_acc.append(relat(exact_markowitz, approx_markowitz))

            DATA[f'{n_assets} - {n_bits}'] = {
                'exact': exact / LOOPS,
                'ising': np.mean(ising_acc),
                'markowitz': np.mean(markowitz_acc)
            }   

with open('accuracy_analysis.json', 'w') as f:
    json.dump(DATA, f)  