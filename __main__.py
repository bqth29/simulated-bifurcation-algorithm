from simulated_bifurcation import simulated_bifurcation
from charts import pie, table
from csv_reader import csv_to_matrix
from data.assets import assets

alpha = 1
sigma, mu, assets_list = csv_to_matrix('./data/cov.csv', './data/mu.csv')

try:

    portfolio = simulated_bifurcation(sigma, mu, alpha)

    print(f"The optimal portfolio is {portfolio}.")

    # Visualization (depending on the number of assets)

    if len([i for i in range(len(portfolio)) if portfolio[i] > 10**(-3)]) > 50:

        table(portfolio, assets_list)

    else:

        pie(portfolio, assets_list)

except:

    print("An error occured")