from models.Partitioning import NumberPartioning, Clique
from models.Inequalities import Knapsack
from models.Covering import VertexCover
import random as rd
import numpy as np

# liste = [2, 5, -1, 3, 8, 4, 2, 17, -32, 6, 11, 4, 5, 7, 8, 9, 1, 2]
# liste = [rd.randint(-10, 10) for _ in range(80)]
# # liste = [rd.random() for _ in range(250)]

# model = NumberPartioning(liste)
# model.optimize(ballistic=False)
# print(model)


# n = 5
# weights = [rd.randint(1, 10) for _ in range(n)]
# costs = [rd.randint(1, 30) for _ in range(n)]

# print(list(zip(weights, costs)))

# model = Knapsack(
#     weights,
#     costs,
#     100
# )
# model.optimize()
# print(model.to_keep)
# print(model.weight_load)
# print(model.total_cost)

# graph = np.array(
#     [
#         [0, 1, 1, 0, 1, 1],
#         [1, 0, 1, 0, 1, 1],
#         [1, 1, 0, 1, 0, 1],
#         [0, 0, 1, 0, 0, 1],
#         [1, 1, 0, 0, 0, 0],
#         [1, 1, 1, 1, 0, 0],
#     ]
# )
# graph = np.array(
#     [
#         [0, 1, 1, 0, 1, 0, 0],
#         [1, 0, 0, 1, 1, 0, 0],
#         [1, 0, 0, 0, 0, 1, 1],
#         [0, 1, 0, 0, 1, 0, 0],
#         [1, 1, 0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0]
#     ]
# )

rand = np.random.randint(0, 10, (25, 25))
rand = (np.sign((rand + rand.T) / 2 - 7) + 1)/2
rand = rand - np.diag(np.diag(rand))

graph = rand

# graph = np.zeros((8,8))
# graph[0,7] = 1
# graph[7,0] = 1
# graph[1,7] = 1
# graph[7,1] = 1
# for i in range(1,7):
#     graph[i,0] = 1
#     graph[0,i] = 1
#     graph[i,i+1] = 1
#     graph[i+1,i] = 1
#     graph[i,i-1] = 1
#     graph[i-1,i] = 1

model = VertexCover(graph)
print(model)
model.optimize()
a = model.colored
print(a)

model.show()

# model = Clique(graph, 4)
# model.optimize(agents = 100)
# print(model.clique_found)
# print(model.clique_index)
# model.comprehensive_search()
# print(model.clique_found)
# print(model.clique_index)

# rand = np.random.randint(0, 2, (50, 50))
# rand = (np.sign((rand + rand.T) / 2 - .6) + 1)/2
# rand = rand - np.diag(np.diag(rand))

# graph = np.block(
#     [
#         [np.ones((50,50)) - np.eye(50), rand],
#         [rand, rand]
#     ]
# )

# # graph = rand

# model = Clique(graph, 3)
# model.optimize(agents = 100, ballistic=False)
# print(model.clique_found)