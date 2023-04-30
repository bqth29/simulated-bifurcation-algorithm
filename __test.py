import src.simulated_bifurcation as sb
from src.simulated_bifurcation.interface import Binary, Integer
from src.simulated_bifurcation.partitioning import NumberPartioning
from src.simulated_bifurcation.qubo import QUBO as qubo
from src.simulated_bifurcation.markowitz import Markowitz
from random import randint
from torch import Tensor, zeros, round
import pandas as pd
import numpy as np

df = pd.read_csv("mu.csv")

assets = list(df.columns)[1:]
dates = list(df.Date)

covariance_filename = "cov.csv"
expected_return_filename = "mu.csv"

complete_monthly_returns = pd.read_csv(expected_return_filename)
complete_monthly_returns.set_index('Date', inplace = True)

cov = pd.read_csv(covariance_filename)
cov.set_index('Unnamed: 0', inplace = True)

mu = np.expand_dims(complete_monthly_returns[assets].loc[dates[-1]].to_numpy(),1)
sigma = cov[assets].loc[assets].to_numpy()

covariance = sigma
expected_return = mu

# matrix = zeros((100, 100))
# for i in range(100):
#     for j in range(i, 100):
#         matrix[i, j] = randint(-10, 10)

# matrix = .5 * (matrix + matrix.t())

# vector = zeros((100, 1))
# for i in range(100):
#     vector[i, 0] = randint(-10, 10)

# mat = Tensor(matrix)
# solver = Markowitz(mat, vector, 1, 5)

solver = Markowitz(
    Tensor(covariance),
    Tensor(expected_return),
)

solver.optimize()

print(solver.solution)
print(solver.objective_value)