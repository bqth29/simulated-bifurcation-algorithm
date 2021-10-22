from models.Markowitz import Markowitz
from data.data import assets, dates
from time import time

markowitz = Markowitz.from_csv(assets_list = assets[:441], number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(stop_criterion = False, window_size = 50, check_frequency = 5000)

print(markowitz)

markowitz.pie()
markowitz.table()