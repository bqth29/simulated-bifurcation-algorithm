from models.Simulated_Bifurcation import Simulated_Bifurcation
from data.assets import assets

sb = Simulated_Bifurcation(assets_list = assets[:65], number_of_bits = 10)
sb.draw_chart()