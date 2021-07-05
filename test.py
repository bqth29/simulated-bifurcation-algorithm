from models.Hamiltionian import Hamiltonian
from models.Markowitz import Markowitz
from data.data import assets, dates

markowitz = Markowitz.from_csv(assets_list = assets[:20], number_of_bits = 2, date = dates[-10])
markowitz.optimize()

markowitz.pie()
markowitz.table()