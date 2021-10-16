from models.SymplecticEulerScheme import SymplecticEulerScheme
from models.Markowitz import Markowitz
from models.Ising import Ising

markowitz = Markowitz.from_csv()
ising = markowitz.to_Ising()

euler = SymplecticEulerScheme(ising.J, ising.h)
result = euler.run()

print(result)