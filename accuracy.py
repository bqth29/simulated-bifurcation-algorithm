import simulated_bifurcation as sb
from models.Markowitz import Markowitz
import random as rd
from data.data import assets
import json

def get_accuracy(max_assets = 20, max_bits = 10, max_dimension = 20, samples = 100, tol = 1e-6):

    data = {}

    for n_assets in range(3, max_assets + 1):
        for n_bits in range(1, max_bits + 1):

            if n_assets * n_bits <= max_dimension:

                exact = 0
                relative_ising = 0
                relative_markowitz = 0

                for _ in range(samples):

                    markowitz = Markowitz.from_csv(assets_list = rd.sample(assets, n_assets), number_of_bits = n_bits)
                    ising = markowitz.__to_Ising__()

                    ising.comprehensive_search()
                    exact_energy = ising.energy
                    markowitz.__from_Ising__(ising)
                    exact_utility_function = markowitz.utlity_function

                    ising.optimize(agents = 100)
                    sb_energy = ising.energy
                    markowitz.__from_Ising__(ising)
                    sb_utility_function = markowitz.utlity_function

                    energy_gap = abs(exact_energy - sb_energy)
                    utility_function_gap = abs(exact_utility_function - sb_utility_function)

                    if energy_gap < tol: exact += 1

                    if exact_energy != sb_energy:

                        if exact_energy == 0: relative_ising += sb_energy
                        else: relative_ising += energy_gap / abs(exact_energy)

                        if exact_utility_function == 0: relative_markowitz += sb_utility_function
                        else: relative_markowitz += utility_function_gap / abs(exact_utility_function)

                data[f'{n_assets}-{n_bits}'] = {
                    'exact (%)': round(100 * exact / samples, 3),
                    'ising': relative_ising / samples,
                    'markowitz': relative_markowitz / samples
                }

    return data

def save_comparison(data):

    with open('accuracy_without_pressure.json', 'x') as f:
        json.dump(data, f)

data = get_accuracy()
save_comparison(data)