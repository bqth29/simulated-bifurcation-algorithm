from models.Markowitz import Markowitz
from data.data import assets, dates

markowitz = Markowitz.from_csv(assets_list = assets[:441], number_of_bits = 1, date = dates[-1])
markowitz.optimize()

print(markowitz)

markowitz.pie()
markowitz.table()

print(markowitz.gain())