from models.Markowitz import Markowitz
from models.Markowitz import recursive_subportfolio_optimization as RSO
from data.data import assets, dates
from time import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt

markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(
    convergence_threshold = 35,
    sampling_period = 60,
    time_step = 0.01, 
    symplectic_parameter = 2,
    agents = 20,
    ballistic=False,
    heated=True,
    final_pressure=1.,
    pressure_slope=0.1
)

print(markowitz)

# ex = 0
# rel = 0
# for _ in range(100):
#     markowitz = Markowitz.from_csv(assets_list = rd.sample(assets, 5), number_of_bits = 3, date = dates[-1], risk_coefficient = 1)
#     markowitz.comprehensive_search()
#     u_ex = markowitz.utlity_function
#     markowitz.optimize(
#         convergence_threshold = 35,
#         sampling_period = 60,
#         time_step = 0.01,
#         symplectic_parameter = 3,
#         agents = 100,
#         ballistic=False,
#         heated=False,
#         final_pressure=1.
#     )
#     u_app = markowitz.utlity_function
#     if u_ex == 0: diff = u_app
#     else: diff = abs(u_app-u_ex)/u_ex
#     if diff == 0: ex += 1
#     rel += diff
#     print(f'{u_ex} // {u_app}')

# print(ex)
# print(rel/100)

# agents = 500
# ising = markowitz.__to_Ising__()
# J, h = ising.J, ising.h
# n = h.shape[0]
# X = 2 * np.random.randint(0,1,size=(n,agents)) - 1
# #X = np.zeros(h.shape)
# # energies = []
# # utilities = []

# for _ in range(50000):
#     for j in range(agents):
#         i = rd.randint(0, n-1)
#         X[i,j] = np.sign(J[i,:] @ X[:,j] - h[i,:])
# #     ising.ground_state = X
# #     energies.append(ising.energy)
# #     markowitz.__from_Ising__(ising)
# #     utilities.append(markowitz.utlity_function)
# # print(np.max(utilities))
# # T = np.arange(1, 50001)
# # f, (ax1,ax2) = plt.subplots(2, 1)
# # ax1.plot(T, energies)
# # ax2.plot(T,utilities)
# # plt.show()

# print(X.shape)

# energies = np.diag(-.5 * X.T @ J @ X + X.T @ h)
# ground_state = X[:,np.argmin(energies)].reshape(-1, 1)
# ising.ground_state = ground_state
# print(ising.energy)
# markowitz.__from_Ising__(ising)
# print(markowitz.utlity_function)


# markowitz.pie()
# markowitz.table()