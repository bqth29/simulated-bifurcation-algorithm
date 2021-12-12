from models.Markowitz import Markowitz
from data.data import assets, dates

markowitz = Markowitz.from_csv(assets_list = assets[:], number_of_bits = 1, date = dates[-1], risk_coefficient = 1)
markowitz.optimize(
    window_size = 50,
    sampling_period = 70,
    time_step = 0.01,
    symplectic_parameter = 2,
    pressure = lambda t : 0.01 * t
)

print(markowitz)       

markowitz.pie()
markowitz.table()