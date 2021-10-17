from models.SymplecticEulerScheme import SymplecticEulerScheme
from models.Markowitz import Markowitz
from models.Ising import Ising
import numpy as np

b = np.ones((4,1))
print(np.sign(b).T)

a = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
print(a)
a = np.roll(a, -1, axis = 1)
print(a)
var = np.var(a, axis = 1)
print(var)
a[:,-1] = var
print(np.shape(var))
print(a)