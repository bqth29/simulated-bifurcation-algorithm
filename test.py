import numpy as np
from sklearn.datasets import make_spd_matrix
from models.Ising import Ising
from models.Hamiltionian import Hamiltonian
from models.Simulation import Simulation
from models.Markowitz import Markowitz
from data.assets import assets

hamiltonian = Hamiltonian(1, 1, lambda t: 0.01 * t)
parameters = Simulation(0.01, 600, 2)

markowitz = Markowitz(assets_list = assets[:25], number_of_bits=8)
markowitz.from_csv("./data/cov.csv", "./data/mu.csv")
markowitz.optimize(hamiltonian,parameters)
print(markowitz.portfolio)

markowitz.pie()
markowitz.table()